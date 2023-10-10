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

def model_selection(config, model_name, max_epochs, features_dim, target_cols):
    keys = config.keys()
    combinations = it.product(*(config[key] for key in keys))
    param_list = list(combinations)
    print(len(param_list))
    tries = 150

    # Specify the number of random combinations you want to select
    num_random_combinations = tries if len(param_list) > tries else len(param_list)
    # Select random combinations without replacement
    random_combinations = random.sample(param_list, num_random_combinations)

    for i, elem in enumerate(random_combinations):
        results_row = {}
        for k, v in zip(keys, elem):
            results_row[k] = v

        accelerator = Accelerator(gradient_accumulation_steps=16)

        model = RegressorModel(
            name=model_name, fflayers=results_row["fflayers"], ffdropout=results_row["ffdropout"],
            activation_function=results_row["activation_function"], features_dim=features_dim,
            target_cols=target_cols
        ).to(device=accelerator.device)

        trainer = Trainer(model, (train_loader, valid_loader), max_epochs, accelerator, weight_decay=results_row["weight_decay"],
                          base_lr=results_row['base_lr'], max_lr=results_row['max_lr'], step_size_up=results_row['step_size_up'], step_size_down=results_row['step_size_down'])

        start_time = time.time()
        train_losses, val_losses = trainer.fit(verbose=False, patience=8, min_delta=0.004)
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

        csv_filename = 'deberta_model_selection.csv'

        with open(csv_filename, 'a', newline='') as csvfile:
            fieldnames = results_row.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # If the file is empty, write the header row
            if csvfile.tell() == 0:
                writer.writeheader()

            # Write the training information to the CSV file
            writer.writerow(results_row)

    print("model selection completed!")


# Token Length: 867
# 'microsoft/deberta-v3-base'
config = {
    'model': 'bert-base-cased', # name of the model to be downloaded with huggingsface transformers
    'name': 'bert', # name of the model to be displayed on the results
    'max_length': 512,
    'batch_size': 8,  # si può provare ad aumentare ma occhio agli out of memory error
}

features = ['text_word_cnt','text_length','text_stopword_cnt',
            'text_punct_cnt','text_different_word_cnt','text_misspelled_cnt',
            'text_word_ratio','text_length_ratio','text_stopword_ratio','text_punct_ratio','text_different_word_ratio',
            'karp_tfidf_scores','2grams_cnt','3grams_cnt','4grams_cnt']
feature_dim = len(features)
multioutput = True

input_cols = ["prompt_question", "text"]
target_cols = ["content", "wording"]

tokenizer = AutoTokenizer.from_pretrained(config['model'])
multioutput = True
prompt_path = "../data/prompts_train.csv"
train_path = "../data/dataset_bf.zip"
train_data = pd.read_csv(train_path, compression='zip', sep=',', index_col=0)
train_data.reset_index(inplace=True)

prompt_data = pd.read_csv(train_path, compression='zip', sep=',', index_col=0)
prompt_data.reset_index(inplace=True)
model_config = AutoConfig.from_pretrained("bert-base-cased")
encoder = AutoModel.from_pretrained('bert-base-cased', config=model_config)
padded_sequences, attention_masks = [[] for i in range(2)]
def tokenizer_padder(df):
    data = df['prompt_text'].apply(tokenizer.tokenize).apply(tokenizer.convert_tokens_to_ids).to_list(),
    max_batch_len = max(len(sequence) for sequence in data)
    max_len = min(max_batch_len, 512)
    attend, no_attend = 1, 0
    padded_sequences, attention_masks = [[] for _ in range(2)]

    for sequence in data:

        # As discussed above, truncate if exceeds max_len
        new_sequence = list(sequence[:max_len])


        attention_mask = [attend] * len(new_sequence)
        pad_length = max_len - len(new_sequence)
        print(pad_length)

        new_sequence.extend([tokenizer.pad_token_id] * pad_length)
        attention_mask.extend([no_attend] * pad_length)

        outputs = encoder(input_ids=torch.tensor(new_sequence), attention_mask=torch.tensor(attention_mask),
                                output_hidden_states=False)
        last_hidden_state = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        logits = sum_embeddings / sum_mask
        print(logits) 
    
        padded_sequences.append(new_sequence)
        attention_masks.append(attention_mask)

    padded_sequences = torch.tensor(padded_sequences)
    attention_masks = torch.tensor(attention_masks)
    outputs = encoder(input_ids=padded_sequences, attention_mask=attention_masks,
                                output_hidden_states=False)
    last_hidden_state = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    logits = sum_embeddings / sum_mask
    print(logits) 

tokenizer_padder(prompt_data)
# prompt_data['text_embedding'] = prompt_data['prompt_text'].apply(lambda x: embedding_creator(x))

# train_df, valid_df = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data["prompt_id"])
# train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df["prompt_id"])
# train_df = oversample_df(train_df)

# train_set = SmartBatchingDataset(df=train_df, tokenizer=tokenizer, input_cols=input_cols, target_cols=target_cols, features_cols=features)
# valid_set = SmartBatchingDataset(df=valid_df, tokenizer=tokenizer, input_cols=input_cols, target_cols=target_cols, features_cols=features)
# test_set = SmartBatchingDataset(df=test_df, tokenizer=tokenizer, input_cols=input_cols, target_cols=target_cols, features_cols=features)
# train_loader = train_set.get_dataloader(batch_size=config['batch_size'], max_len=config['max_length'],pad_id=tokenizer.pad_token_id)
# valid_loader = valid_set.get_dataloader(batch_size=config['batch_size'], max_len=config['max_length'],pad_id=tokenizer.pad_token_id)
# test_loader = test_set.get_dataloader(batch_size=config['batch_size'], max_len=config['max_length'],pad_id=tokenizer.pad_token_id)

# max_epochs = 100
# model_selection_config = {
#     'base_lr': [1e-8, 2e-8],
#     'max_lr': [1e-7, 2e-7],
#     'step_size_up': [500,1000],
#     'step_size_down': [1000,1500],
#     'fflayers': [3],
#     'ffdropout': [5e-2],
#     'activation_function': ["relu"],
#     'weight_decay': [0.001, 0.01, 0.03]
# }
# model_selection(model_selection_config, config["model"], max_epochs, features_dim=feature_dim, target_cols=target_cols)
