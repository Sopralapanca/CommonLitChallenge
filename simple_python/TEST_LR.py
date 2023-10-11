import itertools as it
import pandas as pd
import numpy as np
import random
import time
import json
import csv
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"]="true"

import torch
import torch.nn as nn
from torch.utils.data import Sampler, Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, AutoConfig
from accelerate import Accelerator
from sklearn.model_selection import train_test_split

from TRAINER_utils import oversample_df, evaluate, SmartBatchingDataset, RegressorModel, Trainer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model_config = AutoConfig.from_pretrained("bert-base-cased")
encoder = AutoModel.from_pretrained('bert-base-cased', config=model_config)
def embedder(row):
    paragraphs = row.split('\n')
    even_p = int(len(paragraphs)/2)+1
    center = row.index(paragraphs[even_p])
    first_part = row[:center]
    second_part = row[center:]
    tokenizer.truncation_side='left'
    first_inputs = tokenizer(first_part, return_tensors="pt", truncation=True,
                             max_length=256, return_token_type_ids=False)
    tokenizer.truncation_side='right'
    second_inputs = tokenizer(second_part, return_tensors="pt", truncation=True,
                             max_length=256, return_token_type_ids=False)
    
    input_ids = torch.cat((first_inputs['input_ids'], second_inputs['input_ids']), 1)
    attention_mask = torch.tensor([1]*(input_ids.shape[1])).reshape(1, 512)
    outputs = encoder(input_ids=input_ids, attention_mask=attention_mask,
                                  output_hidden_states=False)   
    last_hidden_state = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    logits = sum_embeddings / sum_mask
    score = 0
    for i, logit in enumerate(logits):
        score += logit ** (i*0.07)
    return score

def model_selection(config, model_name, max_epochs, features_dim, target_cols):
    keys = config.keys()
    combinations = it.product(*(config[key] for key in keys))
    param_list = list(combinations)

    tries = 150

    # Specify the number of random combinations you want to select
    num_random_combinations = tries if len(param_list) > tries else len(param_list)
    # Select random combinations without replacement
    random_combinations = random.sample(param_list, num_random_combinations)

    for i, elem in enumerate(random_combinations):
        results_row = {}
        for k, v in zip(keys, elem):
            results_row[k] = v

        accelerator = Accelerator(gradient_accumulation_steps=4)

        model = RegressorModel(
            name=model_name, fflayers=results_row["fflayers"], ffdropout=results_row["ffdropout"],
            activation_function=results_row["activation_function"], features_dim=features_dim,
            target_cols=target_cols
        ).to(device=accelerator.device)

        trainer = Trainer(model, (train_loader, valid_loader), max_epochs, accelerator, weight_decay=results_row["weight_decay"],
                          lr=results_row['base_lr'])

        start_time = time.time()
        train_losses, val_losses = trainer.fit(verbose=False, patience=5, min_delta=0.004)
        elapsed_time = time.time() - start_time

        id = i

        results_row['id'] = id
        results_row['model-name'] = model_name
        results_row['train-loss'] = train_losses["loss"][-1]
        results_row['valid-loss'] = val_losses["loss"][-1]
        results_row['train-content-loss'] = train_losses["content"][-1].item()
        results_row['valid-content-loss'] = val_losses["content"][-1].item()
        results_row['train-wording-loss'] = train_losses["wording"][-1].item()
        results_row['valid-wording-loss'] = val_losses["wording"][-1].item()
        results_row['best-valid-loss'] = min(val_losses["loss"])
        results_row['elapsed-time'] = elapsed_time
        results_row['epochs'] = len(train_losses["loss"])

        csv_filename = 'bert__siamese_model__selection.csv'

        with open(csv_filename, 'a', newline='') as csvfile:
            fieldnames = results_row.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # If the file is empty, write the header row
            if csvfile.tell() == 0:
                writer.writeheader()

            # Write the training information to the CSV file
            writer.writerow(results_row)

    print("model selection completed!")

# 'microsoft/deberta-v3-base'
config = {
    'model': 'bert-base-cased', # name of the model to be downloaded with huggingsface transformers
    'name': 'bert', # name of the model to be displayed on the results
    'max_length': 512,
    'batch_size': 8,  # si può provare ad aumentare ma occhio agli out of memory error
}

tokenizer = AutoTokenizer.from_pretrained(config['model'])
multioutput = True

train_path = "/storagenfs/c.peluso5/CommonLitChallenge/data/dataset_bf.zip"
train_data = pd.read_csv(train_path, compression='zip', sep=',', index_col=0)
train_data.reset_index(inplace=True)

prompt_path = "/storagenfs/c.peluso5/CommonLitChallenge/data/prompts_train.csv"
prompt_data = pd.read_csv(prompt_path, sep=',', index_col=0)
prompt_data.reset_index(inplace=True)

prompt_data['embeddings'] = prompt_data['prompt_text'].apply(embedder)
train_data = train_data.merge(prompt_data.drop(columns=['prompt_question', 'prompt_text', 'prompt_title']), on='prompt_id')
train_data.drop(columns=['prompt_text'], inplace=True)

features = ['text_word_cnt','text_length','text_stopword_cnt',
            'text_punct_cnt','text_different_word_cnt','text_misspelled_cnt',
            'text_word_ratio','text_length_ratio','text_stopword_ratio','text_punct_ratio','text_different_word_ratio',
            'karp_tfidf_scores','2grams_cnt','3grams_cnt','4grams_cnt', 'embeddings']
feature_dim = len(features)
multioutput = True

input_cols = ["prompt_question", "text"]
target_cols = ["content", "wording"]


train_df, valid_df = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data["prompt_id"])
train_df = oversample_df(train_df)

train_set = SmartBatchingDataset(df=train_df, tokenizer=tokenizer, input_cols=input_cols, target_cols=target_cols, features_cols=features)
valid_set = SmartBatchingDataset(df=valid_df, tokenizer=tokenizer, input_cols=input_cols, target_cols=target_cols, features_cols=features)

train_loader = train_set.get_dataloader(batch_size=config['batch_size'], max_len=config['max_length'],pad_id=tokenizer.pad_token_id)
valid_loader = valid_set.get_dataloader(batch_size=config['batch_size'], max_len=config['max_length'],pad_id=tokenizer.pad_token_id)

max_epochs = 100

model_selection_config = {
    'base_lr': [7e-4, 75e-5, 8e-4],
    # 'max_lr': [1e-7, 2e-7],
    # 'step_size_up': [500,1000],
    # 'step_size_down': [1000,1500],
    'fflayers': [3, 4],
    'ffdropout': [5e-2, 4e-2, 3e-2],
    'activation_function': ["relu"],
    'weight_decay': [4e-2, 3e-2, 2e-2]
}
model_selection(model_selection_config, config["model"], max_epochs, features_dim=feature_dim, target_cols=target_cols)
