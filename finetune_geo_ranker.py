import sys
import os
import random
from transformers import Trainer, TrainingArguments
import torch
from datasets import load_dataset
import argparse
import torch.nn as nn
from datetime import datetime
from peft import get_peft_model, LoraConfig, TaskType
from utils.geo_ranker import get_vlm_for_sequence_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings('ignore')

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from geopy.distance import geodesic
from PIL import Image
import io
import base64
import tarfile
import pickle
import torch

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class MultiModalDataCollator:
    def __init__(self, processor, group_size, args, image_data_path='xxx/dataset/mp16-pro/mp-16-images.tar', member_info_path='xxx/dataset/mp16-pro/tar_index.pkl'):
        self.processor = processor
        self.group_size = group_size
        self.tokenizer = processor.tokenizer
        self.tokenizer.padding_side = 'left'

        self.image_data_path = image_data_path

        self.args = args

        with open(member_info_path, 'rb') as f:
            self.tar_index = pickle.load(f)

    def __call__(self, examples):
        messages_lis = []
        labels = []
        distance_lis = []

        with tarfile.open(self.image_data_path, 'r') as tar_obj:
            for example in examples:
                image_id = example['img_id']
                image = tar_obj.extractfile(self.tar_index[image_id])
                image_data = Image.open(image)

                buffered = io.BytesIO()
                image_data.save(buffered, format="JPEG")
                base64_encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
                image_str = f"data:image/jpeg;base64,{base64_encoded}"

                gps_ground_truth = example['gps']
                gps_ref = example['ref_gps']


                distance = []
                
                for idx, (lat, lon) in enumerate(gps_ref[1:self.group_size+1]):
                    ref_image_id = example['ref_img_id'][idx+1]
                    ref_image = tar_obj.extractfile(self.tar_index[ref_image_id])
                    ref_image_data = Image.open(ref_image)
                    ref_buffered = io.BytesIO()
                    ref_image_data.save(ref_buffered, format="JPEG")
                    ref_base64_encoded = base64.b64encode(ref_buffered.getvalue()).decode('utf-8')
                    ref_image_str = f"data:image/jpeg;base64,{ref_base64_encoded}"

                    # negative sampling
                    gps_neg_ref = gps_ref[-5:]
                    gps_neg_ref_texts = example['ref_texts'][-5:]

                    neg_samples = []
                    for (neg_lat, neg_lon), texts in zip(gps_neg_ref, gps_neg_ref_texts):
                        neg_text = f"latitude: {neg_lat}, longitude: {neg_lon}, {' '.join(texts)}"
                        neg_samples.append(neg_text)

                    ref_texts = ' '.join(example['ref_texts'][idx+1])
                    messages = [{
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_str},
                                {"type": "text", "text": f"How far is this place from latitude: {lat}, longitude: {lon}, {ref_texts}?"},
                                {"type": "image", "image": ref_image_str},
                                {"type": "text", "text": f"Negative examples: {'; '.join(neg_samples)}"},
                            ],
                        }]
                    messages_lis.append(messages)

                    distance.append(geodesic(gps_ground_truth, (lat,lon)).km)
                    distance_lis.append(geodesic(gps_ground_truth, (lat,lon)).km)
                
                label = distance.index(min(distance))
                labels.append(label)

        texts = [self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_lis]
        image_inputs, video_inputs = process_vision_info(messages_lis)

        tokenized_inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        tokenized_inputs['labels'] = torch.tensor(labels)
        tokenized_inputs['distance'] = torch.tensor(distance_lis)

        return tokenized_inputs

class PRMTrainer(Trainer):
    def __init__(self, model=None, huggingface_args=None, aux_args=None, data_collator=None, train_dataset=None, eval_dataset=None, tokenizer=None, compute_metrics=None, preprocess_logits_for_metrics=None):
        super().__init__(model=model, args=huggingface_args,data_collator=data_collator,train_dataset=train_dataset,eval_dataset=eval_dataset,tokenizer=tokenizer,compute_metrics=compute_metrics, preprocess_logits_for_metrics=preprocess_logits_for_metrics)
        self.aux_args = aux_args
        self.loss_type = aux_args.loss_type
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.group_size = aux_args.group_size # positive + negative
        self.pair_wise_loss_fn = nn.MarginRankingLoss(margin=0.0, reduction='mean')

    # overlap with the original compute_loss
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        model_inputs = {k: v for k, v in inputs.items() if k not in ['labels','distance']}
        rewards, outputs = model(**model_inputs, return_output=True)
        rewards = rewards.view(-1, self.group_size)
        distances = inputs['distance'].view(-1, self.group_size)
        
        loss1 = self.pl_loss(rewards, distances, topn=self.aux_args.K)
        loss2 = self.second_order_pl_loss(rewards, distances, topn=self.aux_args.K)
        loss = self.aux_args.lambda1 * loss1 + (1.0-self.aux_args.lambda1) * loss2

        return (loss, (loss, rewards)) if return_outputs else loss

    def pl_loss(self, rewards, distances, topn='all'):
        sorted_indices = torch.argsort(distances, dim=1)
        sorted_rewards = torch.gather(rewards, 1, sorted_indices)

        exp_rewards = torch.exp(sorted_rewards)

        denominator = torch.flip(torch.cumsum(torch.flip(exp_rewards, dims=[1]), dim=1), dims=[1])

        if topn == 'all':
            pass
        else:
            exp_rewards = exp_rewards[:, :topn]
            denominator = denominator[:, :topn]

        log_probs = torch.log(exp_rewards) - torch.log(denominator)

        loss = - torch.mean(log_probs, dim=1)
        loss = torch.mean(loss)
        return loss

    def second_order_pl_loss(self, rewards, distances, topn='all'):
        B, G = rewards.shape

        sort_idx = torch.argsort(distances, dim=1)  # (B, G)
        sorted_rewards = torch.gather(rewards, 1, sort_idx)
        sorted_distances = torch.gather(distances, 1, sort_idx)

        idx_i, idx_j = torch.triu_indices(G, G, offset=1)
        delta_rewards = sorted_rewards[:, idx_i] - sorted_rewards[:, idx_j]  # (B, P)
        delta_distances = sorted_distances[:, idx_i] - sorted_distances[:, idx_j]  # (B, P)

        if topn == 'all':
            pass
        else:
            num = sum(G - i - 1 for i in range(topn))
            delta_rewards = delta_rewards[:, :num]
            delta_distances = delta_distances[:, :num]

        sorted_delta_distances, second_order_idx = torch.sort(delta_distances, dim=1)  # (B, P)
        sorted_delta_rewards = torch.gather(delta_rewards, 1, second_order_idx)  # (B, P)

        exp_scores = torch.exp(sorted_delta_rewards)
        denom = torch.flip(torch.cumsum(torch.flip(exp_scores, dims=[1]), dim=1), dims=[1])
        log_probs = torch.log(exp_scores) - torch.log(denom)
        loss = - torch.mean(log_probs, dim=1).mean()

        return loss



class VLMFinetuner(object):
    def __init__(self, args):
        self.args = args
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.per_device_eval_batch_size = args.per_device_eval_batch_size
        self.total_batch_size = args.total_batch_size
        self.learning_rate = args.learning_rate
        self.server = args.server

        now = datetime.now()
        self.datetime_str = now.strftime("%m%d%H%M")

        self.model_path = args.model_path
        self.continue_ckpt_path = args.continue_ckpt_path
        self.model_save_path = args.model_save_path
        self.tokenizer_save_path = self.model_save_path

        # training init
        self.__model_init__()
        print('model initialized')
        self.__data_init__()
        print('data initialized')
        self.__auto_trainer_init__()
        print('trainer initialized')

        print('Trainable parameters:')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"{name}")

        self.auto_train()
    def __model_init__(self):
        # loading model
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        self.lora_config = LoraConfig(
            r=self.args.lora_r, # rank
            lora_alpha=2*self.args.lora_r, # alpha scaling factor
            lora_dropout=0.05, # dropout rate
            target_modules = ["q_proj", "k_proj", "v_proj"], # target modules
            modules_to_save=["value_head"],
        )

        self.model = get_vlm_for_sequence_regression(
            base_model_name_or_path=self.model_path,
            model_name_or_path=self.model_path,
            model_type="reward",
            lora_config=self.lora_config,
            normalize_reward=False,
            use_flash_attention_2=True,
            ds_config=None,
            init_value_head=True,
            value_head_prefix="value_head",
            device_map=None,
            packing_samples=False,
            lora_path=None,
        )

        # self.model.gradient_checkpointing_enable()
        
    def __data_init__(self):
        self.dataset = load_dataset("parquet", data_files="path_to_file")
        self.tokenized_datasets = self.dataset['train'].select(range(self.args.num_samples))
        self.data_collator = MultiModalDataCollator(processor=self.processor, group_size=self.args.group_size, args=self.args)

    def __auto_trainer_init__(self):

        BATCH_SIZE = args.total_batch_size
        GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // args.per_device_train_batch_size
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size != 1
        if ddp:
            GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

        print(f"world_size: {world_size}")
        print(f"ddp: {ddp}")

        output_path = args.model_save_path

        # Training arguments
        self.training_args = TrainingArguments(
            output_dir=output_path,
            # evaluation_strategy="no",  # Evaluate at the end of each epoch
            eval_steps=10000,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            num_train_epochs=self.args.num_train_epochs,
            weight_decay=0.01,
            logging_strategy="steps",
            logging_steps=5,
            save_strategy=self.args.save_strategy,
            save_steps=0.1,
            save_total_limit=None,
            # fp16=True,
            bf16=True,
            report_to=self.args.report_to,
            logging_dir="./logs",
            dataloader_num_workers=2,
            deepspeed="xxx/deepspeed_config/zero2.json",
            ddp_find_unused_parameters=False,
            metric_for_best_model="mse",
            greater_is_better=False,
            label_names=["labels"],
            remove_unused_columns=False,
            # gradient_checkpointing=True,
            # dataloader_pin_memory=False,
            # dataloader_prefetch_factor=1,
            # max_grad_norm=1.0,
            # seed=42,
        )
        # Initialize the Trainer
        self.trainer = PRMTrainer(
            model=self.model,
            huggingface_args=self.training_args,
            aux_args=self.args,
            train_dataset=self.tokenized_datasets,
            eval_dataset=None,  # Replace with a validation set if available
            data_collator=self.data_collator,
            tokenizer=self.processor.tokenizer,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
            compute_metrics=self.compute_metrics_regression,
        )

    def auto_train(self):
        self.model.train()
        if self.args.continue_ckpt_path:
            self.trainer.train(resume_from_checkpoint = self.args.continue_ckpt_path)
        else:
            self.trainer.train()
        # self.model.save_pretrained(self.model_save_path)
        # self.tokenizer.save_pretrained(self.tokenizer_save_path)



if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank passed from distributed launcher")
    parser.add_argument("--model_path", type=str, default=f"Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--loss_type", type=str, default='mse')
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--total_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--server", type=str, default='1')
    parser.add_argument("--privileged", type=str2bool, default=False)
    parser.add_argument("--report_to", type=str, default='wandb', choices=['none', 'wandb'])
    parser.add_argument("--continue_train", type=str2bool, default=False)
    parser.add_argument("--test_only", type=str2bool, default=False)
    parser.add_argument("--continue_ckpt_path", type=str, default=None)
    parser.add_argument("--model_save_path", type=str, default='xxx')
    parser.add_argument("--dataset", type=str, default='xxx/xxx')
    parser.add_argument("--group_size", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--save_strategy", type=str, default='steps', choices=['no', 'steps', 'epoch'])
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lambda1", type=float, default=0.7)
    parser.add_argument("--K", type=str, default='1')
    args = parser.parse_args()

    if args.K.isdigit():
        args.K = int(args.K)

    finetuner = VLMFinetuner(args)