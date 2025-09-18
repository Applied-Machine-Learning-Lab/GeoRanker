import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import torch
import argparse
import pandas as pd
from tqdm import tqdm
from geopy.distance import geodesic
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel, LoraConfig
import os
from utils.geo_ranker import get_vlm_for_sequence_regression
import numpy as np
import io
import base64
from PIL import Image
import tarfile
import pickle

def evaluate(args):
    if args.dataset == 'im2gps3k':
        data = pd.read_csv('xxx/dataset/im2gps3k/im2gps3k.csv')
    elif args.dataset == 'yfcc4k':
        data = pd.read_csv('xxx/dataset/yfcc4k/yfcc4k.csv')

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj"],
        modules_to_save=["value_head"],
    )

    model = get_vlm_for_sequence_regression(
        base_model_name_or_path=args.model,
        model_name_or_path=args.model,
        model_type="reward",
        lora_config=lora_config,
        bf16=True,
        use_flash_attention_2=True,
        device_map="auto",
        lora_path=args.model_path,
    )

    processor = AutoProcessor.from_pretrained(args.model, use_fast=True)

    with open(args.tar_index_path, 'rb') as f:
        tar_index = pickle.load(f)
    tar_obj = tarfile.open(args.image_data_path)

    messages = []
    ref_gps_all = []
    for i in tqdm(range(data.shape[0]), desc='Processing messages'):
        ref_gps_lis = []
        sample_messages = []

        candidate_gps_lis = np.array(eval(data.loc[i, 'candidate_gps_lis']))[:args.topn]
        candidate_text_lis = np.array(eval(data.loc[i, 'candidate_text_lis']))[:args.topn]
        candidate_text_lis = [[texts[1], texts[3], texts[5]] for texts in candidate_text_lis]
        candidate_img_lis = np.array(eval(data.loc[i, 'candidate_img_lis']))[:args.topn]


        zs_gps_lis = np.array(eval(data.loc[i, 'zs_gps_lis']))[:args.topn_zs]


        main_image_str = f"file://{args.image_path}{data.iloc[i]['IMG_ID']}"


        neg_candidate_gps_lis = np.array(eval(data.loc[i, 'candidate_gps_lis']))[-5:]
        neg_candidate_text_lis = np.array(eval(data.loc[i, 'candidate_text_lis']))[-5:]
        neg_candidate_text_lis = [[texts[1], texts[3], texts[5]] for texts in neg_candidate_text_lis]
        neg_samples = []
        for neg_gps, neg_texts in zip(neg_candidate_gps_lis, neg_candidate_text_lis):
            neg_text = f"latitude: {neg_gps[0]}, longitude: {neg_gps[1]}, {' '.join([elem for elem in neg_texts if elem])}"
            neg_samples.append(neg_text)


        for j in range(args.topn):
            gps = candidate_gps_lis[j]
            ref_texts = ' '.join([elem for elem in candidate_text_lis[j] if elem])
            ref_gps_lis.append(gps)
            ref_image = tar_obj.extractfile(tar_index[candidate_img_lis[j]])
            ref_image_data = Image.open(ref_image)
            ref_buffered = io.BytesIO()
            ref_image_data.save(ref_buffered, format="JPEG")
            ref_image_str = f"data:image/jpeg;base64,{base64.b64encode(ref_buffered.getvalue()).decode('utf-8')}"
            message_content = [
                {"type": "image", "image": main_image_str},
                {"type": "text", "text": f"How far is this place from latitude: {gps[0]}, longitude: {gps[1]}, {ref_texts}?"},
                {"type": "image", "image": ref_image_str},
                {"type": "text", "text": f"Negative examples: {'; '.join(neg_samples)}"},
            ]
            message = [{"role": "user", "content": message_content}]
            sample_messages.append(message)


        for k in range(args.topn_zs):
            zs_gps = zs_gps_lis[k]
            ref_gps_lis.append(zs_gps)
            message_content = [
                {"type": "image", "image": main_image_str},
                {"type": "text", "text": f"How far is this place from latitude: {zs_gps[0]}, longitude: {zs_gps[1]}?"},
                {"type": "text", "text": f"Negative examples: {'; '.join(neg_samples)}"},
            ]
            message = [{"role": "user", "content": message_content}]
            sample_messages.append(message)

        messages.extend(sample_messages)
        ref_gps_all.append(ref_gps_lis)


    all_rewards = []
    for i in tqdm(range(0, len(messages), BSZ), desc='Computing rewards'):
        batch_messages = messages[i:i + BSZ]
        text = processor.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
        processor.tokenizer.padding_side = 'left'
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            rewards, _ = model(**inputs, return_output=True)
        all_rewards.extend(rewards.cpu().tolist())


    for i in tqdm(range(data.shape[0]), desc='Selecting predictions'):
        start_idx = i * (args.topn + args.topn_zs)
        end_idx = start_idx + (args.topn + args.topn_zs)
        sample_rewards = all_rewards[start_idx:end_idx]
        sample_ref_gps = ref_gps_all[i]
        max_reward_idx = sample_rewards.index(max(sample_rewards))
        predicted_gps = sample_ref_gps[max_reward_idx]
        data.loc[i, 'LAT_pred'] = predicted_gps[0]
        data.loc[i, 'LON_pred'] = predicted_gps[1]
        data.loc[i, 'max_reward'] = sample_rewards[max_reward_idx]


    data['geodesic'] = data.apply(lambda x: geodesic((x['LAT'], x['LON']), (x['LAT_pred'], x['LON_pred'])).km, axis=1)


    print('2500km level: ', data[data['geodesic'] < 2500].shape[0] / data.shape[0])
    print('750km level: ', data[data['geodesic'] < 750].shape[0] / data.shape[0])
    print('200km level: ', data[data['geodesic'] < 200].shape[0] / data.shape[0])
    print('25km level: ', data[data['geodesic'] < 25].shape[0] / data.shape[0])
    print('1km level: ', data[data['geodesic'] < 1].shape[0] / data.shape[0])

if __name__ == '__main__':

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2-VL-7B-Instruct', help='base model name')
    parser.add_argument('--model_path', type=str, default=None, help='path to the trained LoRA checkpoint')
    parser.add_argument('--dataset', type=str, default='im2gps3k', help='dataset name')
    parser.add_argument('--image_path', type=str, default='dataset/im2gps3k/im2gps3ktest/', help='path to the main images directory')
    parser.add_argument('--image_data_path', type=str, default='xxx/dataset/mp16-pro/mp-16-images.tar', help='path to the tar file containing candidate images')
    parser.add_argument('--tar_index_path', type=str, default='xxx/dataset/mp16-pro/tar_index.pkl', help='path to the tar index file')
    parser.add_argument('--topn', type=int, default=10, help='number of regular candidates to consider')
    parser.add_argument('--topn_zs', type=int, default=5, help='number of zero-shot candidates to consider')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for evaluation')
    args = parser.parse_args()

    BSZ = args.batch_size

    evaluate(args)