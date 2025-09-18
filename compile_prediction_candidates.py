import pandas as pd
import numpy as np

retrieval_df = pd.read_csv('path_to_retrieval_df.csv')
zs_df = pd.read_csv('path_to_lvlm_zs_predict.csv')
mp16 = pd.read_csv('path_to_MP16_Pro_filtered.csv')
I = np.load('path_to_I.npy') # please refer to G3 for the I.npy
# I_reverse = np.load('path_to_I_reverse.npy') # please refer to G3 for the I_reverse.npy

import re
def extract_location_answer(output_str):
    number_pattern = r'[-+]?\d+\.\d+'
    numbers = re.findall(number_pattern, output_str)
    numbers = numbers[-2:]
    
    if len(numbers) == 2:
        latitude = float(numbers[0])
        longitude = float(numbers[1])
        if latitude < -90 or latitude > 90 or longitude < -180 or longitude > 180:
            return 0.0, 0.0
        return latitude, longitude
    
    return 0.0, 0.0

from tqdm import tqdm
retrieval_df['candidate_gps_lis'], retrieval_df['candidate_text_lis'] = None, None
retrieval_df['reverse_gps_lis'], retrieval_df['reverse_text_lis'] = None, None
retrieval_df['zs_gps_lis'], retrieval_df['zs_text_lis'] = None, None
retrieval_df['candidate_img_lis'], retrieval_df['reverse_img_lis'] = None, None
for i in tqdm(range(retrieval_df.shape[0])):
    candidate_gps_lis = []
    reverse_gps_lis = []
    zs_gps_lis = []

    for j in range(20): 
        col_name = f'candidate_{j}_gps'
        if col_name in retrieval_df.columns and pd.notna(retrieval_df.iloc[i][col_name]):
            lat, lon = extract_location_answer(retrieval_df.iloc[i][col_name])
            candidate_gps_lis.append((lat, lon))
    
    for j in range(20):
        col_name = f'reverse_{j}_gps'
        if col_name in retrieval_df.columns and pd.notna(retrieval_df.iloc[i][col_name]):
            lat, lon = extract_location_answer(retrieval_df.iloc[i][col_name])
            reverse_gps_lis.append((lat, lon))

    response_list = eval(zs_df.iloc[i]['response'])
    for resp in response_list:
        lat, lon = extract_location_answer(resp)
        zs_gps_lis.append((lat, lon))

    candidate_text_lis, candidate_img_lis = [],[]
    for j in range(len(candidate_gps_lis)):
        idx = I[i][j]
        neighbourhood,city,county,state,region,country = mp16.iloc[idx][['neighbourhood', 'city','county', 'state', 'region', 'country']].fillna('').values
        candidate_text_lis.append([neighbourhood,city,county,state,region,country])
        img_id = mp16.iloc[idx]['IMG_ID']
        candidate_img_lis.append(img_id)

    # reverse_text_lis, reverse_img_lis = [],[]
    # for j in range(len(reverse_gps_lis)):
    #     idx = I_reverse[i][j]
    #     neighbourhood,city,county,state,region,country = mp16.iloc[idx][['neighbourhood', 'city','county', 'state', 'region', 'country']].fillna('').values
    #     reverse_text_lis.append([neighbourhood,city,county,state,region,country])
    #     img_id = mp16.iloc[idx]['IMG_ID']
    #     reverse_img_lis.append(img_id)

    retrieval_df.loc[i,'candidate_gps_lis'] = str(candidate_gps_lis)
    retrieval_df.loc[i,'candidate_text_lis'] = str(candidate_text_lis)
    retrieval_df.loc[i,'reverse_gps_lis'] = str(reverse_gps_lis)
    # retrieval_df.loc[i,'reverse_text_lis'] = str(reverse_text_lis)
    retrieval_df.loc[i,'zs_gps_lis'] = str(zs_gps_lis)
    retrieval_df.loc[i,'candidate_img_lis'] = str(candidate_img_lis)
    # retrieval_df.loc[i,'reverse_img_lis'] = str(reverse_img_lis)

retrieval_df.drop(['candidate_0_gps','candidate_1_gps', 'candidate_2_gps', 'candidate_3_gps',
       'candidate_4_gps', 'candidate_5_gps', 'candidate_6_gps',
       'candidate_7_gps', 'candidate_8_gps', 'candidate_9_gps',
       'candidate_10_gps', 'candidate_11_gps', 'candidate_12_gps',
       'candidate_13_gps', 'candidate_14_gps', 'candidate_15_gps',
       'candidate_16_gps', 'candidate_17_gps', 'candidate_18_gps',
       'candidate_19_gps', 'reverse_0_gps', 'reverse_1_gps', 'reverse_2_gps',
       'reverse_3_gps', 'reverse_4_gps', 'reverse_5_gps', 'reverse_6_gps',
       'reverse_7_gps', 'reverse_8_gps', 'reverse_9_gps', 'reverse_10_gps',
       'reverse_11_gps', 'reverse_12_gps', 'reverse_13_gps', 'reverse_14_gps',
       'reverse_15_gps', 'reverse_16_gps', 'reverse_17_gps', 'reverse_18_gps',
       'reverse_19_gps'], axis=1, inplace=True)

retrieval_df.to_csv('xxx/dataset/im2gps3k/im2gps3k.csv', index=False)