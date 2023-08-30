from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from utils.dynamic_padding import SmartBatchingDataset

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords



nltk.download('stopwords')
nltk.download('wordnet')
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def preprocessText(text):
    # replace newline with space
    text = text.replace("\n", "")

    text = text.replace('\r', '')
    # Replace curly apostrophe with straight single quote
    text = text.replace('’', "'")

    # Normalize spaces around punctuation marks
    text = text.strip()

    # lower case
    text = text.lower()

    # split text
    words = text.split()

    # stop word removal
    words = [w for w in words if not w in stop_words]

    # stemming
    # words = [stemmer.stem(w) for w in words]

    # lemmatization
    words = [lemmatizer.lemmatize(w) for w in words]

    text = ' '.join(words)

    return text


def oversample_df(df):
    """

    :param df: Dataframe to be overampled based on prompt_id
    :return: Dataframe oversampled
    """
    classes = df["prompt_id"].value_counts().to_dict()
    most = max(classes.values())
    classes_list = []
    for key in classes:
        classes_list.append(df[df["prompt_id"] == key])
    classes_sample = []
    for i in range(1, len(classes_list)):
        classes_sample.append(classes_list[i].sample(most, replace=True))
    df_maybe = pd.concat(classes_sample)
    final_df = pd.concat([df_maybe, classes_list[0]], axis=0)

    return final_df


def pipeline(config, input_cols, target_cols, dynamic_padding, features, split=0.2, oversample=False):
    """
    :param config:          model configuration dictionary
    :param input_cols:      list of strings defining the columns of the dataframe that needs to be input of the LLM
    :param target_cols:     list of strings defining the columns of the dataframe for the target
    :param dynamic_padding:
    :param features:        list of strings defining the columns of the dataframe that rapresent the added feature to be concatenated as input to the feedforward net
    :param split:           float, percentage of the size of the validation set and test set
    :param oversample:      boolean, if true the training set will be oversampled
    :return:                training, validation, test loaders and the tokenizer used
    """

    data_path = "./data/dataset.csv"
    training_data = pd.read_csv(data_path, sep=',', index_col=0)

    training_data.reset_index(inplace=True)

    train, test = train_test_split(training_data, test_size=split, random_state=42, stratify=training_data["prompt_id"])
    train, valid = train_test_split(train, test_size=split, random_state=42, stratify=train["prompt_id"])

    if oversample:
        train = oversample_df(train)

    tokenizer = AutoTokenizer.from_pretrained(config['model'])

    if dynamic_padding:
        train_set = SmartBatchingDataset(train, tokenizer, input_cols, target_cols, features_cols=features)
        valid_set = SmartBatchingDataset(valid, tokenizer, input_cols, target_cols, features_cols=features)
        test_set = SmartBatchingDataset(test, tokenizer, input_cols, target_cols, features_cols=features)

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
