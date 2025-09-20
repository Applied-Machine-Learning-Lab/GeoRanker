import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import torch
import numpy as np
import io
import base64
from PIL import Image
import tarfile
import pickle
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel, LoraConfig
from utils.geo_ranker import get_vlm_for_sequence_regression
from geopy.distance import geodesic

data_sample = {'IMG_ID': '103433117_266c57c2e6_29_73293249@N00.jpg', 'LAT': np.float64(46.167286), 'LON': np.float64(7.099698), 'candidate_gps_lis': [(46.470601, 11.623621), (46.623701, 12.299613), (47.32975, 11.259956), (46.527441, 12.029183), (44.173955, 6.697883), (46.435313, 11.665763), (46.378019, 11.577111), (44.392579, 7.046012), (44.242555, 6.711788), (46.592254, 12.014923), (44.495158, 6.921644), (39.989944, 20.771198), (46.627631, 12.342967), (46.321741, 11.500024), (46.484387, 9.716248), (46.6329, 12.307619), (42.692908, 0.639953), (42.45747, 13.663215), (46.5987, 11.761575), (46.612463, 12.296475)], 'candidate_text_lis': [['', '', '', 'Trentino-Alto Adige/Südtirol', '', 'Italy'], ['', '', '', 'Trentino-Alto Adige/Südtirol', '', 'Italy'], ['', '', 'Bezirk Innsbruck-Land', '', '', 'Austria'], ['', "Cortina d'Ampezzo", 'Belluno', 'Veneto', '', 'Italy'], ['', '', 'Maritime Alps', "Provence-Alpes-Côte d'Azur", 'Metropolitan France', 'France'], ['', '', 'Provincia di Trento', 'Trentino-Alto Adige/Südtirol', '', 'Italy'], ['', '', 'Provincia di Trento', 'Trentino-Alto Adige/Südtirol', '', 'Italy'], ['', '', 'Cuneo', 'Piedmont', '', 'Italy'], ['', '', 'Alpes-de-Haute-Provence', "Provence-Alpes-Côte d'Azur", 'Metropolitan France', 'France'], ['', 'Marèo - Enneberg - Marebbe', 'South Tyrol', 'Trentino-Alto Adige/Südtirol', '', 'Italy'], ['', '', 'Cuneo', 'Piedmont', '', 'Italy'], ['', '', 'Ioannina Regional Unit', 'Epirus and Western Macedonia', '', 'Greece'], ['', 'Auronzo di Cadore', 'Belluno', 'Veneto', '', 'Italy'], ['', '', 'Provincia di Trento', 'Trentino-Alto Adige/Südtirol', '', 'Italy'], ['', 'Surses', 'Albula', 'Grisons', '', 'Switzerland'], ['', '', 'South Tyrol', 'Trentino-Alto Adige/Südtirol', '', 'Italy'], ['', '', 'La Ribagorza', 'Aragon', '', 'Spain'], ['', '', 'Teramo', 'Abruzzo', '', 'Italy'], ['', '', 'South Tyrol', 'Trentino-Alto Adige/Südtirol', '', 'Italy'], ['', 'Auronzo di Cadore', 'Belluno', 'Veneto', '', 'Italy']], 'reverse_gps_lis': [(44.933261, -123.043807), (40.774333, -73.7705), (40.6725, -74.018666), (36.071536, -94.175003), (33.391229, -111.876697), (42.341711, -71.085464), (22.724829, 75.880209), (45.53375, -122.691927), (33.584039, -111.949434), (41.591056, -93.621474), (22.19168, 84.5782), (35.515927, -97.627689), (41.591056, -93.621474), (42.345544, -71.159179), (39.705073, -75.120391), (38.922453, -77.036232), (40.817069, -74.210956), (39.49133, -74.531678), (33.381083, -111.757428), (33.503506, -111.927702)], 'reverse_text_lis': [['', 'Salem', 'Marion County', 'Oregon', '', 'United States'], ['Queens County', 'New York', '', 'New York', '', 'United States'], ['Brooklyn', 'New York', '', 'New York', '', 'United States'], ['', 'Fayetteville', 'Washington County', 'Arkansas', '', 'United States'], ['', 'Mesa', 'Maricopa County', 'Arizona', '', 'United States'], ['Fenway / Kenmore', 'Boston', 'Suffolk County', 'Massachusetts', '', 'United States'], ['Indore City', 'Indore', 'Juni Indore Tahsil', 'Madhya Pradesh', '', 'India'], ['Northwest District', 'Portland', 'Multnomah County', 'Oregon', '', 'United States'], ['', 'Scottsdale', 'Maricopa County', 'Arizona', '', 'United States'], ['Downtown', 'Des Moines', 'Polk County', 'Iowa', '', 'United States'], ['', 'Rajgangpur', 'Rajagangapur', 'Odisha', '', 'India'], ['', 'Bethany', 'Oklahoma County', 'Oklahoma', '', 'United States'], ['Downtown', 'Des Moines', 'Polk County', 'Iowa', '', 'United States'], ['Brighton', 'Boston', 'Suffolk County', 'Massachusetts', '', 'United States'], ['', 'Glassboro', 'Gloucester County', 'New Jersey', '', 'United States'], ['Ward 1', 'Washington', '', 'District of Columbia', '', 'United States'], ['', 'Montclair', 'Essex County', 'New Jersey', '', 'United States'], ['', 'Galloway Township', 'Atlantic County', 'New Jersey', '', 'United States'], ['', 'Mesa', 'Maricopa County', 'Arizona', '', 'United States'], ['5th Avenue Shops & Boutiques', 'Scottsdale', 'Maricopa County', 'Arizona', '', 'United States']], 'candidate_img_lis': ['39_19_3845791320.jpg', '8e_cc_8846800462.jpg', 'f5_f2_3691652602.jpg', 'ad_92_9529891144.jpg', 'e5_5c_5856211104.jpg', '72_56_3758272909.jpg', '5d_f9_7499978086.jpg', 'd6_bb_5991611744.jpg', '97_3a_9014613871.jpg', '58_b5_7708792906.jpg', '0f_e3_9647092523.jpg', '51_62_4598330915.jpg', '8a_ed_4337418008.jpg', '36_be_3877419599.jpg', '80_44_6144637934.jpg', '40_95_3904966178.jpg', '62_f0_231401125.jpg', 'be_6d_4262687722.jpg', '48_84_8217169912.jpg', 'fa_7e_3904941222.jpg'], 'reverse_img_lis': ['3b_dc_8233181983.jpg', '19_91_3682612234.jpg', '0a_0d_8457964042.jpg', 'b2_85_3820790344.jpg', '1c_35_12255335684.jpg', '65_02_9721447696.jpg', 'b2_61_12098158003.jpg', '59_df_13136203875.jpg', '1c_e6_8167844479.jpg', '32_e7_3557503635.jpg', 'f3_91_11993222293.jpg', '66_c7_9265061848.jpg', 'dd_a3_3558134562.jpg', '9b_22_12684959904.jpg', '5a_db_5934394636.jpg', 'b0_6d_4389972809.jpg', 'd2_1b_9706245101.jpg', 'ed_30_8117073641.jpg', 'ee_2a_11124595144.jpg', 'cc_a3_11346914904.jpg']}

# Configuration - Adjust these paths as needed for your environment
CONFIG = {
    'model': 'Qwen/Qwen2-VL-7B-Instruct',  # Base model name
    'model_path': 'rootpath/checkpoints/test',  # Path to trained LoRA checkpoint; set to None if using base model
    'image_path': 'rootpath/dataset/im2gps3k/im2gps3ktest/',  # Path to main images directory
    'image_data_path': 'rootpath/dataset/mp16-pro/mp-16-images.tar',  # Path to tar file with candidate images
    'tar_index_path': 'rootpath/dataset/mp16-pro/tar_index.pkl',  # Path to tar index pickle
    'topn': 10,  # Number of regular candidates
}

def quick_start_inference(config):
    """
    Quick start inference for a single sample.
    Builds messages for candidates and zero-shot GPS, computes rewards using the model,
    selects the GPS with max reward, and computes geodesic distance to ground truth.
    """
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

    # Load tar file and index for candidate images
    with open(config['tar_index_path'], 'rb') as f:
        tar_index = pickle.load(f)
    tar_obj = tarfile.open(config['image_data_path'])

    # Extract sample data
    ref_gps_lis = []
    sample_messages = []

    # Regular candidates
    candidate_gps_lis = np.array(data_sample['candidate_gps_lis'])[:config['topn']]
    candidate_text_lis = np.array(data_sample['candidate_text_lis'])[:config['topn']]
    candidate_text_lis = [[texts[1], texts[3], texts[5]] for texts in candidate_text_lis]
    candidate_img_lis = np.array(data_sample['candidate_img_lis'])[:config['topn']]

    # Main image path
    main_image_str = f"file://{config['image_path']}{data_sample['IMG_ID']}"

    # Generate negative examples (last 5 from candidates)
    neg_candidate_gps_lis = np.array(data_sample['candidate_gps_lis'])[-5:]
    neg_candidate_text_lis = np.array(data_sample['candidate_text_lis'])[-5:]
    neg_candidate_text_lis = [[texts[1], texts[3], texts[5]] for texts in neg_candidate_text_lis]
    neg_samples = []
    for neg_gps, neg_texts in zip(neg_candidate_gps_lis, neg_candidate_text_lis):
        neg_text = f"latitude: {neg_gps[0]}, longitude: {neg_gps[1]}, {' '.join([elem for elem in neg_texts if elem])}"
        neg_samples.append(neg_text)
    neg_text_str = f"Negative examples: {'; '.join(neg_samples)}"

    # Process regular candidates
    for j in range(config['topn']):
        gps = candidate_gps_lis[j]
        ref_texts = ' '.join([elem for elem in candidate_text_lis[j] if elem])
        ref_gps_lis.append(gps)

        # Load and encode reference image
        ref_image = tar_obj.extractfile(tar_index[candidate_img_lis[j]])
        ref_image_data = Image.open(ref_image)
        ref_buffered = io.BytesIO()
        ref_image_data.save(ref_buffered, format="JPEG")
        ref_image_str = f"data:image/jpeg;base64,{base64.b64encode(ref_buffered.getvalue()).decode('utf-8')}"

        message_content = [
            {"type": "image", "image": main_image_str},
            {"type": "text", "text": f"How far is this place from latitude: {gps[0]}, longitude: {gps[1]}, {ref_texts}?"},
            {"type": "image", "image": ref_image_str},
            {"type": "text", "text": neg_text_str},
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

    # Compute geodesic distance to ground truth
    gt_gps = (data_sample['LAT'], data_sample['LON'])
    pred_gps = (predicted_gps[0], predicted_gps[1])
    geodesic_dist_km = geodesic(gt_gps, pred_gps).km

    # Output results
    print("Rewards for all candidates:")
    for idx, (reward, gps) in enumerate(zip(all_rewards, ref_gps_lis)):
        print(f"  Candidate {idx}: GPS=({gps[0]:.6f}, {gps[1]:.6f}), Reward={reward:.4f}")
    
    print(f"\nPredicted GPS (max reward): ({predicted_gps[0]:.6f}, {predicted_gps[1]:.6f}) with reward {max_reward:.4f}")
    print(f"Ground Truth GPS: ({data_sample['LAT']:.6f}, {data_sample['LON']:.6f})")
    print(f"Geodesic Distance: {geodesic_dist_km:.2f} km")

    return predicted_gps, max_reward, geodesic_dist_km

if __name__ == '__main__':
    # Run quick start inference
    predicted_gps, max_reward, dist = quick_start_inference(CONFIG)
    print(f"\nQuick Start Complete! Predicted location is {dist:.2f} km from ground truth.")


# Rewards for all candidates:
#   Candidate 0: GPS=(46.470601, 11.623621), Reward=-6.8438
#   Candidate 1: GPS=(46.623701, 12.299613), Reward=-6.6562
#   Candidate 2: GPS=(47.329750, 11.259956), Reward=-7.2812
#   Candidate 3: GPS=(46.527441, 12.029183), Reward=-7.1562
#   Candidate 4: GPS=(44.173955, 6.697883), Reward=-4.1250
#   Candidate 5: GPS=(46.435313, 11.665763), Reward=-6.7188
#   Candidate 6: GPS=(46.378019, 11.577111), Reward=-6.5312
#   Candidate 7: GPS=(44.392579, 7.046012), Reward=-5.0938
#   Candidate 8: GPS=(44.242555, 6.711788), Reward=-4.3438
#   Candidate 9: GPS=(46.592254, 12.014923), Reward=-6.8750

# Predicted GPS (max reward): (44.173955, 6.697883) with reward -4.1250
# Ground Truth GPS: (46.167286, 7.099698)
# Geodesic Distance: 223.77 km

# Quick Start Complete! Predicted location is 223.77 km from ground truth.