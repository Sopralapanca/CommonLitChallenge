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


def pipeline(config, split=0.2, only_text=False):
    summaries_train_path = "./data/summaries_train.csv"
    prompt_train_path = "./data/prompts_train.csv"

    train_data = pd.read_csv(summaries_train_path, sep=',', index_col=0)
    prompt_data = pd.read_csv(prompt_train_path, sep=',', index_col=0)

    training_data = train_data.merge(prompt_data, on='prompt_id')

    #string_columns = ["text", "prompt_question", "prompt_title", "prompt_text"]

    #for col in string_columns:
        # apply preprocessText function to each text column in the dfTrain dataframe
    #    training_data[col] = training_data[col].apply(lambda x: preprocessText(x))

    #summary_word_count = training_data['text'].apply(lambda x: len(x.split()))
    #prompt_word_count = training_data['prompt_text'].apply(lambda x: len(x.split()))
    #print("Summary word count before preprocessing: ", summary_word_count.max())
    #print("prompt word count before preprocessing: ", prompt_word_count.max())

    training_data["text"] = training_data["text"].apply(lambda x: preprocessText(x))
    training_data["prompt_text"] = training_data["prompt_text"].apply(lambda x: preprocessText(x, intense=True))

    #summary_word_count = training_data['text'].apply(lambda x: len(x.split()))
    #prompt_word_count = training_data['prompt_text'].apply(lambda x: len(x.split()))
    #print("Summary word count after preprocessing: ", summary_word_count.max())
    #print("prompt word count after preprocessing: ", prompt_word_count.max())
    #print(training_data["prompt_text"][0])

    train, test = train_test_split(training_data, test_size=split)
    train, valid = train_test_split(train, test_size=split)

    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    if only_text:
        train_encodings = tokenizer(train["text"].values.tolist(), truncation=True, padding=True,
                                    add_special_tokens=True)
        val_encodings = tokenizer(valid["text"].values.tolist(), truncation=True, padding=True, add_special_tokens=True)
        test_encodings = tokenizer(test["text"].values.tolist(), truncation=True, padding=True, add_special_tokens=True)

    else:
        # combine strings of two dataframe columns and output as a list
        train_encodings = tokenizer(train["text"].values.tolist(), train["prompt_text"].values.tolist(),
                                    truncation=True, padding=True, add_special_tokens=True)
        val_encodings = tokenizer(valid["text"].values.tolist(), valid["prompt_text"].values.tolist(),
                                    truncation=True, padding=True, add_special_tokens=True)
        test_encodings = tokenizer(test["text"].values.tolist(), test["prompt_text"].values.tolist(),
                                    truncation=True, padding=True, add_special_tokens=True)


    target_cols = ["content", "wording"]
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
