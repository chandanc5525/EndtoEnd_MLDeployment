# Importing Basic Python Libraries
import numpy as np 
import pandas as pd


# Write a function to load the dataset using pandas function
def load_data():
    data = pd.read_csv('rent_apartments.csv')
    return data

# Assign get_data() to df
df = load_data()


# print(df)