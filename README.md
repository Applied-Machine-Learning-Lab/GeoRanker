# GeoRanker: Distance-Aware Ranking for Worldwide Image Geolocalization

This is the code, checkpoint, and dataset repository for `GeoRanker: Distance-Aware Ranking for Worldwide Image Geolocalization`

## Environment

```bash
conda create -n georanker python=3.11
conda activate georanker

# Addtional modules
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils[decord]==0.0.8
pip install pandas geopy
pip install flash-attn --no-build-isolation
pip install scikit-learn deepspeed datasets peft torchvision wandb
conda install mpi4py
```

**❗❗❗We also uploaded the YAML file for our environment, but the Transformers version used during our experiments was `4.52.0.dev0`. You may want to set it to a newer official version when running this repository.**

## Quick Start

### Run with your images (calculating rewards between a query and some candidates)

please first modify the image paths, candidate_gps_lis, gt_lat, and gt_lon in `quick_start.py` file, then run `python quick_start.py` to check the rewards and prediction.

### Run with sampled im2gps3k data

You will need to first download the **mp16-pro** tar file and the **tar_index.pkl** file from [Hugging Face](https://huggingface.co/datasets/Jia-py/MP16-Pro). Additionally, please download the IM2GPS3K image dataset as described in the *Dataset* section. Then, modify the relevant paths in **quick_start_im2gps3k.py** and run `python quick_start_im2gps3k.py`.

## Dataset

### Evaluation Datasets

IM2GPS3K: [images](http://www.mediafire.com/file/7ht7sn78q27o9we/im2gps3ktest.zip) and [metadata](https://raw.githubusercontent.com/TIBHannover/GeoEstimation/original_tf/meta/im2gps3k_places365.csv); YFCC4K: [images](http://www.mediafire.com/file/3og8y3o6c9de3ye/yfcc4k.zip) and [metadata](https://github.com/TIBHannover/GeoEstimation/releases/download/pytorch/yfcc25600_places365.csv); MP16-Pro: [Huggingface](https://huggingface.co/datasets/Jia-py/MP16-Pro)

You can also find the meta data for IM2GPS3K, YFCC4K, retrieval checkpoints of G3, retrieval index in [Huggingface](https://huggingface.co/Jia-py/G3-checkpoint)

### GeoRanking Dataset

We have uploaded the dataset to `dataset/georanking`

```python
dataset = load_dataset("parquet", data_files="path_to_file", split="train")

>>> dataset
Dataset({
    features: ['img_id', 'gps', 'ref_gps', 'ref_img_id', 'ref_texts'],
    num_rows: 100000
})
```

* img_id: ID of query image in MP16-Pro dataset
* gps: gps of query image
* ref_gps: gps list for candidates
* ref_img_id: image id list for candidates
* ref_texts: textual descriptions list for candidates

## Checkpoints

The lora weights are put under `checkpoints/`.

## File Structure

```
.
├── checkpoints/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── dataset/
│   ├── im2gps3k/
│   │   ├── im2gps3k.csv
│   │   ├── im2gps3k_metadata_and_images_should_be_put_here
│   │   └── I.npy -> retrieval index results for im2gps3k
│   ├── mp16-pro/
│   │   └── mp16-pro_metadata_and_images_and_should_be_put_here
│   └── yfcc4k/
│       ├── yfcc4k.csv
│       ├── yfcc4k_metadata_and_images_should_be_put_here
│       └── I.npy -> retrieval index results for yfcc4k
├── deepspeed_config/
│   └── zero2.json
├── utils/
│   └── geo_ranker.py -> main file for georanker
├── compile_prediction_candidates.py -> compile retrieval and generated candidates to one file
├── evaluate.py
├── finetune_geo_ranker.py -> script for training georanker
├── environment.yml
└── lvlm_zs_predict.py -> script for generating candidates with lvlm
```

For MP16-Pro dataset, please refer to [G3](https://arxiv.org/pdf/2405.14702).

## Running

1. Training GeoRanker

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus 4 finetune_geo_ranker.py --model_path=Qwen/Qwen2-VL-7B-Instruct --model_save_path=xxx --group_size=7
   ```
2. Generating candidates with LVLM

   ```bash
   python lvlm_zs_predict.py --api_key=sk-xxx --model_name=xxx --base_url=xxx --root_path=xxx/dataset/yfcc4k
   ```
3. Compiling generated and retrieval candidates to one file (we have uploaded the retrieval candidates and generated candidates for IM2GPS3K and YFCC4K under dataset folder). `We have uploaded the index file I.npy for IM2GPS3K and YFCC4K.`

   ```bash
   python compile_prediction_candidates.py
   ```
4. Evaluation

   ```bash
   # we recommend using larger batch size during inference
   python evaluate.py --model_path=path_to_lora --dataset=im2gps3k --topn=12 --topn_zs=3 --batch_size=16
   ```
