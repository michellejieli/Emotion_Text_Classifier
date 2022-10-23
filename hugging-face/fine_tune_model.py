#!/usr/bin/env python

"""
Fine Tuning emotion text classifier with Friends show data 

Based on official tutorial
"""

from transformers import AutoTokenizer
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import pandas as pd
import json

# Load the dataset
with open("friends.json") as f:
    data = json.load(f)

# convert to dataframe
df = pd.DataFrame(data)

df = df.stack().reset_index()
df = df.drop(['level_0', 'level_1'], axis=1)
print(df) 

# get utterance and label
utterances = [d.get('utterance') for d in df[0]]
# label 
labels = [d.get('emotion') for d in df[0]]

# create new training_data

data_tuples = list(zip(utterances,labels))

train_data = pd.DataFrame(data_tuples, columns=['sentence','label'])
print(train_data.head())
