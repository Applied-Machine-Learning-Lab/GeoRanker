import sys
import os
import warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from datasets import load_dataset
import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
from flash_attn.utils.distributed import all_gather
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig
import argparse
import os
from utils import *
from datetime import datetime

from peft import PeftModel
from peft import get_peft_model, LoraConfig, TaskType

from transformers import  Qwen2VLForConditionalGeneration

def get_vlm_for_sequence_regression(base_model_name_or_path: str, model_name_or_path: str, model_type: str,
*,
bf16=True, lora_config=None, normalize_reward=False, use_flash_attention_2=False, ds_config: dict=None, init_value_head: bool=False, value_head_prefix="value_head",device_map=None,packing_samples=False,lora_path=None,**kwargs) -> nn.Module:
    config = AutoConfig.from_pretrained(base_model_name_or_path, trust_remote_code=True)
    config.normalize_reward = normalize_reward
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

    cls_class = _get_reward_model(Qwen2VLForConditionalGeneration, value_head_prefix, packing_samples)

    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    model = cls_class.from_pretrained(model_name_or_path, config=config, trust_remote_code=True, torch_dtype = torch.bfloat16 if bf16 else "auto", quantization_config=None, device_map=device_map, **kwargs)

    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
        model.eval()
    elif lora_config is not None:
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)

    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        print("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True
    
    model.config.use_cache = False

    if init_value_head:
        value_head = getattr(model, value_head_prefix)
        if dschf is not None:
            with deepspeed.zero.GatheredParameters([value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    return model

def _get_reward_model(base_pretrained_model, value_head_prefix="value_head", packing_samples=False):
    class RewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            self.value_head = nn.Linear(config.hidden_size, 1, bias=False)
            self.packing_samples = packing_samples
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(self, input_ids, attention_mask=None, pixel_values=None, return_output=False, **kwargs):

            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_hidden_states=True,
                **kwargs
            )
            last_hidden_states = outputs.hidden_states[-1]

            values = self.value_head(last_hidden_states).squeeze(-1)
            reward = values[:, -1]
            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std
            return (reward, outputs) if return_output else reward

    return RewardModel