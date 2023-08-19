import spacy
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import KFold

from utils.dynamic_padding import SmartBatchingDataset, SmartBatchingSampler, SmartBatchingCollate


def preprocessText(text, intense=False):
    try:
        # replace newline with space
        text = text.replace("\n", " ")

        text = text.replace('\r', '')
        # Replace curly apostrophe with straight single quote
        text = text.replace('’', "'")

        # Normalize spaces around punctuation marks
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s([.,!?])', r'\1', text)
        text = re.sub(r'([.,!?])\s', r'\1', text)
        text = text.strip()

        if intense:
            # lower case
            text = text.lower()

            # remove punctuations
            translator = str.maketrans("", "", string.punctuation)
            text = text.translate(translator)

            # split text
            words = text.split()

            # stop word removal
            stop_words = spacy.lang.en.stop_words.STOP_WORDS
            words = [w for w in words if not w in stop_words]

            # stemming
            stemmer = PorterStemmer()
            words = [stemmer.stem(w) for w in words]

            # lemmatization
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(w) for w in words]

            # return pre-processed paragraph text
            text = ' '.join(words)

        return text
    except:
        return text


class CommonLitDataset(Dataset):
    def __init__(self, encodings, labels):
        # Initialize the tokenizer for the desired transformer model
        self.encodings = encodings
        # list of target columns
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        input_ids, attention_mask = self.encodings["input_ids"][idx], self.encodings["attention_mask"][idx]
        target = torch.tensor(self.labels[idx])

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)

        return input_ids, attention_mask, target


def pipeline(config, input_cols, target_cols, dynamic_padding, preprocess_cols=None, split=0.2):
    if preprocess_cols is None:
        preprocess_cols = []

    summaries_train_path = "./data/summaries_train.csv"
    prompt_train_path = "./data/prompts_train.csv"

    train_data = pd.read_csv(summaries_train_path, sep=',', index_col=0)
    prompt_data = pd.read_csv(prompt_train_path, sep=',', index_col=0)

    training_data = train_data.merge(prompt_data, on='prompt_id')

    for col in input_cols:
        # apply preprocessText function to each text column in the dfTrain dataframe
        if col in preprocess_cols:
            training_data[col] = training_data[col].apply(lambda x: preprocessText(x, intense=True))
        else:
            training_data[col] = training_data[col].apply(lambda x: preprocessText(x))

    train, test = train_test_split(training_data, test_size=split, random_state=42)
    train, valid = train_test_split(train, test_size=split, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    print("Tokenization...")
    if dynamic_padding:
        train_set = SmartBatchingDataset(train, tokenizer, input_cols, target_cols)

        valid_set = SmartBatchingDataset(valid, tokenizer, input_cols, target_cols)
        test_set = SmartBatchingDataset(test, tokenizer, input_cols, target_cols)

        train_loader = train_set.get_dataloader(batch_size=config['batch_size'], max_len=config["max_length"],
                                                pad_id=tokenizer.pad_token_id)
        valid_loader = valid_set.get_dataloader(batch_size=config['batch_size'], max_len=config["max_length"],
                                                pad_id=tokenizer.pad_token_id)
        test_loader = test_set.get_dataloader(batch_size=config['batch_size'], max_len=config["max_length"],
                                              pad_id=tokenizer.pad_token_id)

    else:
        input_train_df = train[input_cols]
        # Combine strings from multiple columns with [CLS], [SEP], and [SEP] separators
        input_train_df['combined_col'] = input_train_df.apply(
            lambda row: tokenizer.cls_token + ' ' + f' {tokenizer.sep_token} '.join(row) + f' {tokenizer.sep_token}',
            axis=1)

        input_valid_df = valid[input_cols]
        # Combine strings from multiple columns with [CLS], [SEP], and [SEP] separators
        input_valid_df['combined_col'] = input_valid_df.apply(
            lambda row: tokenizer.cls_token + ' ' + f' {tokenizer.sep_token} '.join(row) + f' {tokenizer.sep_token}',
            axis=1)

        input_test_df = test[input_cols]
        # Combine strings from multiple columns with [CLS], [SEP], and [SEP] separators
        input_test_df['combined_col'] = input_test_df.apply(
            lambda row: tokenizer.cls_token + ' ' + f' {tokenizer.sep_token} '.join(row) + f' {tokenizer.sep_token}',
            axis=1)

        # combine each string in the columns into a single string and tokenize it
        train_encodings = tokenizer(input_train_df.values.tolist(), truncation=True, padding=True)
        val_encodings = tokenizer(input_valid_df.values.tolist(), truncation=True, padding=True)
        test_encodings = tokenizer(input_test_df.values.tolist(), truncation=True, padding=True)

        target_cols_train = train[target_cols].values
        target_cols_valid = valid[target_cols].values
        target_cols_test = test[target_cols].values

        train_set = CommonLitDataset(encodings=train_encodings, labels=target_cols_train)
        valid_set = CommonLitDataset(encodings=val_encodings, labels=target_cols_valid)
        test_set = CommonLitDataset(encodings=test_encodings, labels=target_cols_test)

        train_loader = DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        valid_loader = DataLoader(dataset=valid_set, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        test_loader = DataLoader(dataset=test_set, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    return train_loader, valid_loader, test_loader, tokenizer
