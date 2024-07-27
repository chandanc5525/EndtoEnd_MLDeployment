from collection import load_data
import re
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def prepare_data():
    
    # Steps to be Followed are as follows:
    # 1. Load Dataset 
    data = load_data()
    
    # 2. Encode Dataset
    data_encode = encode_cat_cols(data)
    
    # 3. Data Cleaning
    df = parse_garden_col(data_encode)
    
    return data

def encode_cat_cols(data):
    return pd.get_dummies(data, 
                          columns = ['balcony',
                                    'parking', 
                                    'furnished', 
                                    'garage', 
                                    'storage'], 
                          drop_first=True)


def parse_garden_col(data):
    for i in range(len(data)):
        if data.garden[i]=='Not present':
            data.garden[i]=0
        else: 
            data.garden[i] = int(re.findall(r'\d+', data.garden[i])[0])
    return data 


df = prepare_data()

print(df)