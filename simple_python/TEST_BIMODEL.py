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
from sklearn.model_selection import train_test_split

from TRAINER_utils import oversample_df, SmartBatchingDataset, RegressorModel, Trainer

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"]="true"

def evaluate(test_loader, model, device, trainer):

    preds = []
    running_loss = 0.
    running_loss_content = 0.
    running_loss_wording = 0.
    for el in test_loader:
        input_ids, attention_mask, target = el

        attention_mask = attention_mask.to(device)
        target = target.to(device)

        ids = input_ids[0].to(device)
        inputs = [ids, input_ids[1]]

        output = model(inputs=inputs, attention_mask=attention_mask)
        loss, content_loss, wording_loss = trainer.loss_fn(output, target, multioutput)
        running_loss += loss.item()
        # running_loss_content += content_loss.item()
        # running_loss_wording += wording_loss.item()

        preds.append(output)

        del input_ids, attention_mask, target, loss, ids, inputs, output

    test_loss = running_loss / len(test_loader)
    # test_content_loss = running_loss_content / len(test_loader)
    # test_wording_loss = running_loss_wording / len(test_loader)
    print("Test Loss:", test_loss)
    # print("Test Content Loss:", test_content_loss)
    # print("Test Wording Loss:", test_wording_loss)
    preds = torch.concat(preds)
    return preds, test_loss

config = {
    'model': 'bert-base-cased', # name of the model to be downloaded with huggingsface transformers
    'name': 'bert', # name of the model to be displayed on the results
    'max_length': 512,
    'batch_size': 8,  # si può provare ad aumentare ma occhio agli out of memory error
    'epochs': 100,
    'lr': 8e-4,
    'gradient_accumulation_steps': 16,
    'adam_eps': 1e-6,  # 1e-8 default
    'freeze_encoder': True,
    'fflayers': 3,
    'ffdropout': 0.05,
    'activation_function': 'relu',
    'weight_decay': 0.001
}
accelerator = Accelerator(gradient_accumulation_steps=config['gradient_accumulation_steps'])
tokenizer = AutoTokenizer.from_pretrained(config['model'])
multioutput = False

train_path = "../data/dataset_bf.zip"
train_data = pd.read_csv(train_path, compression='zip', sep=',', index_col=0)
print(train_data.select_dtypes(include=np.number).columns.tolist())
train_data.reset_index(inplace=True)
train_w_df, valid_w_df = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data["prompt_id"])
train_w_df, test_w_df = train_test_split(train_w_df, test_size=0.2, random_state=42, stratify=train_w_df["prompt_id"])
train_w_df = oversample_df(train_w_df)

features_wording = ['text_word_cnt', 
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

wording = ["content"]
feature_dim = len(features_wording)
input_wording = ['prompt_question', 'text']

# Model Construction pooling options == combination , mean-pooling
model_wording = RegressorModel(name=config["model"], fflayers=config["fflayers"], pooling='combination',
                ffdropout=config["ffdropout"],activation_function=config["activation_function"],
                features_dim=feature_dim, target_cols=wording).to(device=accelerator.device)


f = open('test_log.txt', 'a')

name_W = 'WORDING'
train_set = SmartBatchingDataset(df=train_w_df, tokenizer=tokenizer, input_cols=input_wording, target_cols=wording, features_cols=features_wording)
valid_set = SmartBatchingDataset(df=valid_w_df, tokenizer=tokenizer, input_cols=input_wording, target_cols=wording, features_cols=features_wording)
test_set = SmartBatchingDataset(df=test_w_df, tokenizer=tokenizer, input_cols=input_wording, target_cols=wording, features_cols=features_wording)
train_w_loader = train_set.get_dataloader(batch_size=config['batch_size'], max_len=config['max_length'],pad_id=tokenizer.pad_token_id)
valid_w_loader = valid_set.get_dataloader(batch_size=config['batch_size'], max_len=config['max_length'],pad_id=tokenizer.pad_token_id)
test_w_loader = test_set.get_dataloader(batch_size=config['batch_size'], max_len=config['max_length'],pad_id=tokenizer.pad_token_id)
trainer = Trainer(model=model_wording, loaders=(train_w_loader, valid_w_loader), accelerator=accelerator, epochs=config['epochs'],
                        lr=config["lr"], weight_decay=config["weight_decay"], name=name_W)
print(f'Training the WORDING model')
train_losses, val_losses = trainer.fit(multioutput)
info = {'train-loss': train_losses["loss"][-1],
        'valid-loss': val_losses["loss"][-1],
        'best-valid-loss': min(val_losses["loss"])}
print(info)
f.write('\n')
f.write(json.dumps(info))
checkpoint = torch.load(f"./{name_W}_checkpoint.pt")
model_wording.load_state_dict(checkpoint['model_state_dict'])
model_wording.eval()
print(f'Testing the WORDING model')
_, test_loss = evaluate(test_loader=test_w_loader, model=model_wording, trainer=trainer, device=accelerator.device)
f.write(f'\nTest Loss of WORDING: {test_loss}')

train_path = "../data/dataset.zip"
train_data = pd.read_csv(train_path, compression='zip', sep=',', index_col=0)
print(train_data.select_dtypes(include=np.number).columns.tolist())
train_data.reset_index(inplace=True)
train_c_df, valid_c_df = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data["prompt_id"])
train_c_df, test_c_df = train_test_split(train_c_df, test_size=0.2, random_state=42, stratify=train_c_df["prompt_id"])
train_c_df = oversample_df(train_c_df)

features_content = ['distance', 
                    'stop_cnt', 
                    'punctuation_cnt', 
                    'entities', 
                    'text_pos', 
                    'different_word_cnt_ratio', 
                    '2grams_correct_cnt', 
                    '3grams_correct_cnt', 
                    '4grams_correct_cnt', 
                    'misspelled_text_cnt', 
                    'length_ratio', 
                    'tfidf_scores']
content = ["wording"]
feature_dim = len(features_content)
input_content = ['prompt_question', 'corrected_text']

name_C = 'CONTENT'
# Model Construction pooling options == combination , mean-pooling
model_content = RegressorModel(name=config["model"], fflayers=config["fflayers"], pooling='combination',
                ffdropout=config["ffdropout"],activation_function=config["activation_function"],
                features_dim=feature_dim, target_cols=content).to(device=accelerator.device)
train_set = SmartBatchingDataset(df=train_c_df, tokenizer=tokenizer, input_cols=input_content, target_cols=content, features_cols=features_content)
valid_set = SmartBatchingDataset(df=valid_c_df, tokenizer=tokenizer, input_cols=input_content, target_cols=content, features_cols=features_content)
test_set = SmartBatchingDataset(df=test_c_df, tokenizer=tokenizer, input_cols=input_content, target_cols=content, features_cols=features_content)
train_c_loader = train_set.get_dataloader(batch_size=config['batch_size'], max_len=config['max_length'],pad_id=tokenizer.pad_token_id)
valid_c_loader = valid_set.get_dataloader(batch_size=config['batch_size'], max_len=config['max_length'],pad_id=tokenizer.pad_token_id)
test_c_loader = test_set.get_dataloader(batch_size=config['batch_size'], max_len=config['max_length'],pad_id=tokenizer.pad_token_id)
trainer = Trainer(model=model_content, loaders=(train_c_loader, valid_c_loader), accelerator=accelerator, epochs=config['epochs'],
                        lr=config["lr"], weight_decay=config["weight_decay"], name=name_C)
f = open('test_log.txt', 'a')
print(f'Training the CONTENT model')
train_losses, val_losses = trainer.fit(multioutput)
info = {'train-loss': train_losses["loss"][-1],
        'valid-loss': val_losses["loss"][-1],
        'best-valid-loss': min(val_losses["loss"])}
print(info)
f.write('\n')
f.write(json.dumps(info))
checkpoint = torch.load(f"./{name_C}_checkpoint.pt")
model_content.load_state_dict(checkpoint['model_state_dict'])
model_content.eval()
print(f'Testing the CONTENT model')
_, test_loss = evaluate(test_loader=test_c_loader, model=model_content, trainer=trainer, device=accelerator.device)
f.write(f'\nTest Loss of CONTENT: {test_loss}')

print('COMPUTING THE COMBINATION')
preds = []
running_loss = 0.
running_loss_content = 0.
running_loss_wording = 0.
for el_w, el_c in zip(test_w_loader, test_c_loader):
    input_ids, attention_mask, target = el_c
    attention_mask = attention_mask.to(accelerator.device)
    target_c = target.to(accelerator.device)
    ids = input_ids[0].to(accelerator.device)
    inputs = [ids, input_ids[1]]
    content_output = model_content(inputs=inputs, attention_mask=attention_mask)

    input_ids, attention_mask, target = el_w
    attention_mask = attention_mask.to(accelerator.device)
    target_w = target.to(accelerator.device)
    ids = input_ids[0].to(accelerator.device)
    inputs = [ids, input_ids[1]]
    wording_output = model_wording(inputs=inputs, attention_mask=attention_mask)

    target = torch.stack((target_w, target_c), 2)[:,0]
    output = torch.stack((content_output, wording_output), 2)[:,0]
    loss, content_loss, wording_loss = trainer.loss_fn(output, target, True)
    running_loss += loss.item()
    running_loss_content += content_loss.item()
    running_loss_wording += wording_loss.item()

    preds.append(output)

    del input_ids, attention_mask, target, loss, ids, inputs, output

test_loss = running_loss / len(test_c_loader)
test_content_loss = running_loss_content / len(test_c_loader)
test_wording_loss = running_loss_wording / len(test_c_loader)
print("Test Loss:", test_loss)
print("Test Content Loss:", test_content_loss)
print("Test Wording Loss:", test_wording_loss)
f.write(f'\nTest Loss of both models: {test_loss}')

f.close()