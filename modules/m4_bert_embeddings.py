from m1_main import *

import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512, add_special_tokens=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the pooled output as it represents the entire sentence
    embeddings = outputs.pooler_output
    return embeddings

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

# Load parsed dataset
df = pd.read_parquet(parsed_xml_cpc_path)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  # Set the model to evaluation mode

# Stores the embeddings as lists of floats in a new column
df['bert_embeddings'] = df['CTB'].apply(lambda x: get_bert_embedding(x).tolist()[0])

df.to_excel(workflow_folder + r'\excel\bert_embeddings.xlsx')
df.to_parquet(workflow_folder + r'\parquet\bert_embeddings.parquet')