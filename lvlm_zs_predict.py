import requests
import base64
import os
import re
import pandas as pd
import numpy as np
import ast
from pandarallel import pandarallel
from tqdm import tqdm
import json
import time
import argparse

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_response(image_path, base_url, api_key, model_name, detail="low", max_tokens=200, temperature=1.2, n=10):
    base64_image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model_name,
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": """Suppose you are an expert in geo-localization, you have the ability to give two number GPS coordination given an image.
                Please give me the location of the given image.
                Remember, you must have an answer, just output your best guess, don't answer me that you can't give a location.
                Your answer should be in the following JSON format without any other information: {"latitude": float,"longitude": float}.
                Your answer should be in the following JSON format without any other information: {"latitude": float,"longitude": float}."""
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": detail
                }
                }
            ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": n
    }

    response = requests.post(base_url, headers=headers, json=payload, timeout=(30,60))
    ans = []
    for choice in response.json()['choices']:
        try:
            ans.append(choice['message']['content'])
        except:
            ans.append('{"latitude": 0.0,"longitude": 0.0}')
    return ans


def process_row(row, base_url, api_key, model_name, root_path, image_path):
    image_path = os.path.join(root_path, image_path, row["IMG_ID"])
    try:
        response = get_response(image_path, base_url, api_key, model_name)
    except Exception as e:
        response = "None"
        print(e)
    row['response'] = response
    return row

def run(args):
    api_key = args.api_key
    model_name = args.model_name
    base_url = args.base_url
    root_path = args.root_path
    text_path = args.text_path
    image_path = args.image_path
    result_path = args.result_path
    process = args.process

    if os.path.exists(os.path.join(root_path, result_path)):
        df = pd.read_csv(os.path.join(root_path, result_path))
        df_rerun = df[df['response'].isna()]
        print('Need Rerun:', df_rerun.shape[0])
        df_rerun = df_rerun.parallel_apply(lambda row: process_row(row, base_url, api_key, model_name, root_path, image_path), axis=1)
        df.update(df_rerun)
        df.to_csv(os.path.join(root_path, result_path), index=False)
    else:
        df = pd.read_csv(os.path.join(root_path, text_path))
        df = df.parallel_apply(lambda row: process_row(row, base_url, api_key, model_name, root_path, image_path), axis=1)
        df.to_csv(os.path.join(root_path, result_path), index=False)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    model_name = "xxx"
    api_key = "sk-xxx"
    base_url = "xxx"

    root_path = "xxx/dataset/yfcc4k"
    text_path = "yfcc4k_places365.csv"
    image_path = "yfcc4k"
    result_path = "llm_predict_results_zs_{}.csv".format(model_name)

    pandarallel.initialize(progress_bar=True, nb_workers=32)
    args.add_argument('--api_key', type=str, default=api_key)
    args.add_argument('--model_name', type=str, default=model_name)
    args.add_argument('--base_url', type=str, default=base_url)
    args.add_argument('--root_path', type=str, default=root_path)
    args.add_argument('--text_path', type=str, default=text_path)
    args.add_argument('--image_path', type=str, default=image_path)
    args.add_argument('--result_path', type=str, default=result_path)
    args = args.parse_args()
    print(args)

    run(args)


