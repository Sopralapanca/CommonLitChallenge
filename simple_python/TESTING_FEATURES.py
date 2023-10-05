import more_itertools
import pandas as pd
import numpy as np
import itertools
import json
import copy
import gc
import os

import torch
import torch.nn as nn
from torch.utils.data import Sampler, Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, AutoConfig
from accelerate import Accelerator

from trainer_utils import oversample_df, SmartBatchingDataset, RegressorModel, Trainer

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ["TOKENIZERS_PARALLELISM"]="true"

def evaluate(test_loader, model, device, trainer):

    preds = []
    running_loss = 0.
    for el in test_loader:
        input_ids, attention_mask, target = el

        attention_mask = attention_mask.to(device)
        target = target.to(device)

        ids = input_ids[0].to(device)
        inputs = [ids, input_ids[1]]

        output = model(inputs=inputs, attention_mask=attention_mask)
        loss, content_loss, wording_loss = trainer.loss_fn(output, target, multioutput)
        running_loss += loss.item()

        preds.append(output)

        del input_ids, attention_mask, target, loss, ids, inputs, output

    test_loss = running_loss / len(test_loader)
    print("Test Loss:", test_loss)
    preds = torch.concat(preds)
    return preds, test_loss

config = {
    'model': 'bert-base-cased', # name of the model to be downloaded with huggingsface transformers
    'name': 'bert cased', # name of the model to be displayed on the results
    'max_length': 512,
    'batch_size': 8,  # si può provare ad aumentare ma occhio agli out of memory error
    'epochs': 40,
    'lr': 7e-4,
    'gradient_accumulation_steps': 16,
    'adam_eps': 1e-6,  # 1e-8 default
    'freeze_encoder': True,
    'fflayers': 3,
    'ffdropout': 0.05,
    'activation_function': 'relu',
    'weight_decay': 0.001
}

train_path = "./CommonLitChallenge/data/dataset_bf.zip"
train_data = pd.read_csv(train_path, compression='zip', sep=',', index_col=0)
print(train_data.select_dtypes(include=np.number).columns.tolist())
train_data.reset_index(inplace=True)
accelerator = Accelerator(gradient_accumulation_steps=config['gradient_accumulation_steps'])
tokenizer = AutoTokenizer.from_pretrained(config['model'])
features = [
    'text_word_cnt', 
    'text_length', 
    'text_stopword_cnt', 
    'text_punct_cnt', 
    'text_different_word_cnt', 
    'text_misspelled_cnt', 
    'text_word_ratio', 
    'text_length_ratio', 
    'text_stopword_ratio', 
    'text_punct_ratio', 
    'text_different_word_ratio', 
    'karp_tfidf_scores', 
    '2grams_cnt', 
    '3grams_cnt', 
    '4grams_cnt']
feature_dim = len(features)
target_cols = ["content", "wording"]
input_cols = ['prompt_title', 'text']

multioutput = True

# Model Construction pooling options == combination , mean-pooling
model = RegressorModel(name=config["model"], fflayers=config["fflayers"], pooling='combination',
                ffdropout=config["ffdropout"],activation_function=config["activation_function"],
                features_dim=feature_dim, target_cols=target_cols).to(device=accelerator.device)

from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data["prompt_id"])
train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df["prompt_id"])
train_df = oversample_df(train_df)
print(train_df["prompt_id"].value_counts())

train_set = SmartBatchingDataset(df=train_df, tokenizer=tokenizer, input_cols=input_cols, target_cols=target_cols, features_cols=features)
valid_set = SmartBatchingDataset(df=valid_df, tokenizer=tokenizer, input_cols=input_cols, target_cols=target_cols, features_cols=features)
test_set = SmartBatchingDataset(df=test_df, tokenizer=tokenizer, input_cols=input_cols, target_cols=target_cols, features_cols=features)

train_loader = train_set.get_dataloader(batch_size=config['batch_size'], max_len=config['max_length'],pad_id=tokenizer.pad_token_id)
valid_loader = valid_set.get_dataloader(batch_size=config['batch_size'], max_len=config['max_length'],pad_id=tokenizer.pad_token_id)
test_loader = test_set.get_dataloader(batch_size=config['batch_size'], max_len=config['max_length'],pad_id=tokenizer.pad_token_id)

trainer = Trainer(model=model, loaders=(train_loader, valid_loader), accelerator=accelerator, epochs=config['epochs'],
                        lr=config["lr"], weight_decay=config["weight_decay"])

f = open('test_log.txt', 'a')
train_losses, val_losses = trainer.fit(multioutput)

info = {    
    'train-loss': train_losses["loss"][-1],
    'valid-loss': val_losses["loss"][-1],
    'valid-content-loss': val_losses["content"][-1].item(),
    'valid-wording-loss': val_losses["wording"][-1].item(),
    'best-valid-loss': min(val_losses["loss"]),
    }
print(info)
f.write(json.dumps(info))

checkpoint = torch.load("./checkpoint.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
_, test_loss = evaluate(test_loader=test_loader, model=model, trainer=trainer, device=accelerator.device)

print(f'Test Loss: {test_loss}')
f.write(f'Test Loss: {test_loss}')

f.close()
