import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import torch
import numpy as np
import io
import base64
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig
from utils.geo_ranker import get_vlm_for_sequence_regression
from geopy.distance import geodesic

# Example inputs - Adjust these paths and data for your use case
CONFIG = {
    'model': 'Qwen/Qwen2-VL-7B-Instruct',  # Base model name
    'model_path': './checkpoints/',  # Path to trained LoRA checkpoint; set to None if using base model
    'query_image_path': '/path/to/your/query_image.jpg',  # Path to the main query image
    'candidate_image_paths': [  # List of paths to candidate images (must match len(candidate_gps_lis))
        '/path/to/candidate1.jpg',
        '/path/to/candidate2.jpg',
        # ... add more as needed
    ],
    'candidate_gps_lis': [  # List of (lat, lon) tuples for candidates
        (46.470601, 11.623621),
        (46.623701, 12.299613),
        # ... add more as needed
    ],
    'gt_lat': None,  # Optional: Ground truth latitude for distance calculation
    'gt_lon': None,  # Optional: Ground truth longitude for distance calculation
    'topn': 10,  # Number of candidates to process (min(len(candidate_gps_lis), topn))
}

def load_image_to_base64(image_path):
    """
    Load an image from path and convert to base64 string.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path not found: {image_path}")
    image_data = Image.open(image_path)
    buffered = io.BytesIO()
    image_data.save(buffered, format="JPEG")
    image_str = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
    return image_str

def quick_start_inference(config):
    """
    Quick start inference for a single sample with user-provided query image, candidate images, and GPS.
    Builds messages for candidates, computes rewards using the model, and selects the GPS with max reward.
    Optionally computes geodesic distance if GT provided.
    """
    # Validate inputs
    candidate_gps_lis = config['candidate_gps_lis'][:config['topn']]
    candidate_image_paths = config['candidate_image_paths'][:config['topn']]
    if len(candidate_gps_lis) != len(candidate_image_paths):
        raise ValueError("Number of candidate GPS must match number of candidate image paths.")

    # Load model and processor
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj"],
        modules_to_save=["value_head"],
    )
    model = get_vlm_for_sequence_regression(
        base_model_name_or_path=config['model'],
        model_name_or_path=config['model'],
        model_type="reward",
        lora_config=lora_config,
        bf16=True,
        use_flash_attention_2=True,
        device_map="auto",
        lora_path=config['model_path'],
    )
    processor = AutoProcessor.from_pretrained(config['model'], use_fast=True)

    # Load query image to base64
    main_image_str = load_image_to_base64(config['query_image_path'])

    # Extract reference GPS list
    ref_gps_lis = list(candidate_gps_lis)
    sample_messages = []

    # Process candidates (no text descriptions or negatives, as per user request)
    for j in range(len(candidate_gps_lis)):
        gps = candidate_gps_lis[j]
        ref_gps_lis.append(gps)  # Already appended in list above, but ensure

        # Load and encode candidate reference image
        ref_image_str = load_image_to_base64(candidate_image_paths[j])

        # Simplified message: query image + GPS query text + candidate image
        message_content = [
            {"type": "image", "image": main_image_str},
            {"type": "text", "text": f"How far is this place from latitude: {gps[0]}, longitude: {gps[1]}?"},
            {"type": "image", "image": ref_image_str},
        ]
        message = [{"role": "user", "content": message_content}]
        sample_messages.append(message)

    # Compute rewards for all messages
    all_rewards = []
    for msg in sample_messages:
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        processor.tokenizer.padding_side = 'left'
        image_inputs, video_inputs = process_vision_info(msg)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            rewards, _ = model(**inputs, return_output=True)
        all_rewards.append(rewards.cpu().item())

    # Select prediction: GPS with max reward
    max_reward_idx = np.argmax(all_rewards)
    predicted_gps = ref_gps_lis[max_reward_idx]
    max_reward = all_rewards[max_reward_idx]

    # Compute geodesic distance to ground truth if provided
    geodesic_dist_km = None
    if config['gt_lat'] is not None and config['gt_lon'] is not None:
        gt_gps = (config['gt_lat'], config['gt_lon'])
        pred_gps = (predicted_gps[0], predicted_gps[1])
        geodesic_dist_km = geodesic(gt_gps, pred_gps).km

    # Output results
    print("Rewards for all candidates:")
    for idx, (reward, gps) in enumerate(zip(all_rewards, ref_gps_lis)):
        print(f"  Candidate {idx}: GPS=({gps[0]:.6f}, {gps[1]:.6f}), Reward={reward:.4f}")
    
    print(f"\nPredicted GPS (max reward): ({predicted_gps[0]:.6f}, {predicted_gps[1]:.6f}) with reward {max_reward:.4f}")
    if config['gt_lat'] is not None and config['gt_lon'] is not None:
        print(f"Ground Truth GPS: ({config['gt_lat']:.6f}, {config['gt_lon']:.6f})")
        print(f"Geodesic Distance: {geodesic_dist_km:.2f} km")
    else:
        print("No ground truth provided; skipping distance calculation.")

    return predicted_gps, max_reward, geodesic_dist_km

if __name__ == '__main__':
    # Run quick start inference
    predicted_gps, max_reward, dist = quick_start_inference(CONFIG)
    if dist is not None:
        print(f"\nQuick Start Complete! Predicted location is {dist:.2f} km from ground truth.")
    else:
        print("\nQuick Start Complete!")