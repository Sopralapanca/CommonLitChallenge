import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import os
import gc
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
from spacy.lang.en import English
import string
import spacy
from transformers import BertPreTrainedModel, BertModel
from transformers import AutoConfig, AutoTokenizer
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification,AutoTokenizer
import torch.optim as optim
from transformers import DebertaModel
from transformers import DebertaPreTrainedModel

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchtext
import nltk
from torch.autograd import Variable
from torch import Tensor
import swifter

"""#Get dataset



"""

import re
from nltk.corpus import stopwords
import seaborn as sns
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sklearn.preprocessing import MinMaxScaler
#Load data

# Load data

summaries_train_path = "https://raw.githubusercontent.com/Sopralapanca/CommonLitChallenge/main/data/summaries_train.csv"
prompt_train_path = "https://raw.githubusercontent.com/Sopralapanca/CommonLitChallenge/main/data/prompts_train.csv"

df_train_summaries = pd.read_csv(summaries_train_path, sep=',', index_col=0)
df_train_prompt = pd.read_csv(prompt_train_path, sep=',', index_col=0)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

df_train_summaries['tokenized_text'] = df_train_summaries.swifter.apply(lambda x: tokenizer.tokenize(x.text), axis=1)
df_train_summaries['tokenized_text_len'] = df_train_summaries.swifter.apply(lambda x: len(x.tokenized_text), axis=1)
df_train_summaries.drop(columns=['tokenized_text'], inplace=True)

df_train_prompt['tokenized_prompt_text'] = df_train_prompt.swifter.apply(lambda x: tokenizer.tokenize(x.prompt_text), axis=1)
df_train_prompt['tokenized_prompt_text_len'] = df_train_prompt.swifter.apply(lambda x: len(x.tokenized_prompt_text), axis=1)
df_train_prompt.drop(columns=['tokenized_prompt_text'], inplace=True)

"""# Text processing methods"""
"""# Data preprocessing methods"""
print(" DATA processing")
# normalize the data between 0 and 1 taking into consideration the prompt title

def normalize_col(df, col):
    # Create a Min-Max Scaler
    scaler = MinMaxScaler()
    df[col] = df.groupby('prompt_id')[col].transform(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())

    return df


nltk.download('stopwords')
# get list of all stopwords in english
stopword_list = set(stopwords.words('english'))

from nltk.tokenize import word_tokenize


def preprocessText(text, removal=True):
    # replace newline with space
    text = text.replace("\n", " ")
    text = text.replace('\r', ' ')

    # Normalize spaces around punctuation marks
    text = re.sub(r"[^A-Za-z0-9']", r' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Replace curly apostrophe with straight single quote
    text = text.replace('’', "'")

    # Normalize spaces around punctuation marks
    text = text.strip()

    if removal:
      # lower case
      text = text.lower()

      # split text
      words = text.split()

      # stop word removal
      words = [w for w in words if not w in stopword_list]

      # stemming
      # words = [stemmer.stem(w) for w in words]

      # lemmatization
      #words = [lemmatizer.lemmatize(w) for w in words]

      text = ' '.join(words)

    return text

def add_row(df1, df2, preprocess=False):
    #append row on the head of the dataframe

    row = df2.unique().tolist()[0]

    if preprocess:
        # apply text preprocessing on the row, removing punctuation, stopwrods...
        row = preprocessText(row)

    combined_data = pd.concat([pd.Series([row]),df1.loc[:]]).reset_index(drop=True) #append row on the head of the dataframe

    return combined_data

# Count the stop words in the text.
def count_stopwords(text: str) -> int:
    words = text.split()
    stopwords_count = sum(1 for word in words if word.lower() in stopword_list)
    return stopwords_count

# Count the punctuations in the text.
# punctuation_set -> !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
def count_punctuation(text: str) -> int:
    punctuation_set = set(string.punctuation)
    punctuation_count = sum(1 for char in text if char in punctuation_set)
    return punctuation_count

def different_word_counter(row):
    row = preprocessText(row, removal=False)
    words_list = []
    for w in row.split(' '):
        if w not in words_list:
            words_list.append(w)
    return len(words_list)

def count_words(text):
    return len(text.split(' '))

# ref https://stats.stackexchange.com/questions/570698/should-i-remove-stopwords-before-generating-n-grams
# first find ngrams, then remove the one with stopowrds

from nltk.util import ngrams
nltk.download('punkt')

def count_ngrams(text, n):
    text = preprocessText(text, False)

    keep_ngrams = []
    tokens = word_tokenize(text)
    try:
        n_grams = list(ngrams(tokens, n))
        all_ngrams =  [' '.join(gram) for gram in n_grams]

        # remove ngrams that starts or end with stopwords
        for ngram in all_ngrams:
            if not ngram[0] in stopword_list or not ngram[-1] in stopword_list:
                keep_ngrams.append(ngram)
    except:
        pass

    return keep_ngrams

def count_cooccurring_ngrams(text_ngrams, prompt_text_ngrams):
    # Find the common n-grams between the two columns
    common_ngrams = set(text_ngrams) & set(prompt_text_ngrams)
    return len(common_ngrams)

from sklearn.feature_extraction.text import TfidfVectorizer
import sympy

# Group by 'prompt_id' and compute TF-IDF separately for each class
def compute_TFIDF(df):
    tfidf_vectorizers = {}
    karp_tfidf_scores = {}

    for class_id, group in df.groupby('prompt_id'):
        text_data = group['text_nostop']

        prompt_question_data = group['prompt_question']
        prompt_title_data = group['prompt_title']
        prompt_text_data = group['prompt_text_nostop']

        # Concatenate the preprocessed data for TF-IDF calculation
        combined_data = add_row(text_data, prompt_question_data, True)
        combined_data = add_row(combined_data, prompt_title_data, True)
        combined_data = add_row(combined_data, prompt_text_data)

        # Compute TF-IDF
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(combined_data)
        tfidf_vectorizers[class_id] = {'vectorizer': tfidf_vectorizer, 'matrix': tfidf_matrix}

        tfidf_matrix = tfidf_matrix[3:] #remove first 3 rows f the matrix since they belongs to prompt_text, prompt_question, prompt_title

        modulus = sympy.randprime(tfidf_matrix.shape[1]-100, tfidf_matrix.shape[1])

        # Iterate through documents and calculate TF-IDF scores
        for index, row in group.iterrows():
            doc_tfidf = tfidf_matrix[index - group.index[0]].toarray()[0]

            doc_tfidf = doc_tfidf[doc_tfidf>0]

            gamma = 1e-2
            single_tfidf_score = sum([t**(gamma*i) for i, t in enumerate(doc_tfidf)]) % modulus

            karp_tfidf_scores[index] = single_tfidf_score

    # Add the calculated average TF-IDF scores as a new column to the DataFrame
    df['karp_tfidf_scores'] = [karp_tfidf_scores[index] for index in df.index]

    return df

"""# Preprocessing prompt dataframe before merging"""

def prompt_feature_engineer(dataframe: pd.DataFrame, feature: str = 'prompt_text') -> pd.DataFrame:
    dataframe[f'{feature}_word_cnt'] = dataframe[feature].swifter.apply(count_words)
    dataframe[f'{feature}_length'] = dataframe[feature].swifter.apply(len)
    dataframe[f'{feature}_stopword_cnt'] = dataframe[feature].swifter.apply(count_stopwords)
    dataframe[f'{feature}_punct_cnt'] = dataframe[feature].swifter.apply(count_punctuation)

    dataframe[f'{feature}_different_word_cnt'] = dataframe[feature].swifter.apply(different_word_counter)

    dataframe[f'{feature}_nostop'] = dataframe[feature].swifter.apply(preprocessText)

    # ngrams finding
    for n in range(2,5):
        col = f"{feature}_{n}grams"
        dataframe[col] = dataframe.swifter.apply(lambda row: count_ngrams(row[feature], n), axis=1)


    return dataframe

df_train_prompt = prompt_feature_engineer(df_train_prompt)

"""# Merging dataframe and extract features"""

df_train = df_train_summaries.merge(df_train_prompt, on='prompt_id')

import itertools

word_list = []
for prompt, query, title in zip(df_train_prompt.prompt_text.tolist(), df_train_prompt.prompt_question.tolist(), df_train_prompt.prompt_title.tolist()):
    word_list.append(prompt.replace('\n', ' ').split())
    word_list.append(query.replace('\n', ' ').split())
    word_list.append(title.replace('\n', ' ').split())

token_list = list(itertools.chain(*word_list))

"""Preprocess the data"""

from spellchecker import SpellChecker

spell = SpellChecker()

spell.word_frequency.load_words(token_list)

def misspelled(text):
    words = text.split()
    misspelled = spell.unknown(words)
    return len(misspelled)

import numpy as np

# This function applies all the above preprocessing functions on a text feature.
def feature_engineer(dataframe: pd.DataFrame, feature: str = 'text') -> pd.DataFrame:
    dataframe[f'{feature}_word_cnt'] = dataframe[feature].swifter.apply(count_words)
    dataframe[f'{feature}_length'] = dataframe[feature].swifter.apply(len)
    dataframe[f'{feature}_stopword_cnt'] = dataframe[feature].swifter.apply(count_stopwords)
    dataframe[f'{feature}_punct_cnt'] = dataframe[feature].swifter.apply(count_punctuation)

    # vedere se prima effetturare preprocess del testo
    dataframe[f'{feature}_different_word_cnt'] = dataframe[feature].swifter.apply(different_word_counter)


    # misspelled counter
    dataframe[f'{feature}_misspelled_cnt'] = dataframe[feature].swifter.apply(lambda x: misspelled(x))

    # ratios
    dataframe[f'{feature}_word_ratio'] = dataframe[f'{feature}_word_cnt'] / dataframe['prompt_text_word_cnt']
    dataframe.drop(['prompt_text_word_cnt'], axis=1, inplace=True)

    dataframe[f'{feature}_length_ratio'] = dataframe[f'{feature}_length'] / dataframe['prompt_text_length']
    dataframe.drop(['prompt_text_length'], axis=1, inplace=True)

    dataframe[f'{feature}_stopword_ratio'] = dataframe[f'{feature}_stopword_cnt'] / dataframe['prompt_text_stopword_cnt']
    dataframe.drop(['prompt_text_stopword_cnt'], axis=1, inplace=True)

    dataframe[f'{feature}_punct_ratio'] = dataframe[f'{feature}_punct_cnt'] / dataframe['prompt_text_punct_cnt']
    dataframe.drop(['prompt_text_punct_cnt'], axis=1, inplace=True)


    # vedere se prima effetturare preprocess del testo
    dataframe[f'{feature}_different_word_cnt'] = dataframe[feature].swifter.apply(lambda row: different_word_counter(row))

    dataframe[f'{feature}_different_word_ratio'] = dataframe[f'{feature}_different_word_cnt'] / dataframe['prompt_text_different_word_cnt']
    dataframe.drop(['prompt_text_different_word_cnt'], axis=1, inplace=True)


    dataframe[f'{feature}_nostop'] = dataframe[feature].swifter.apply(preprocessText)
    dataframe = compute_TFIDF(dataframe)

    dataframe.drop(['text_nostop', 'prompt_text_nostop'], axis=1, inplace=True)

    normalize_cols = [f'{feature}_word_cnt',f'{feature}_length',f'{feature}_stopword_cnt',
                      f'{feature}_punct_cnt', f'{feature}_misspelled_cnt',
                     f'{feature}_different_word_cnt', f'{feature}_different_word_ratio',
                     f'{feature}_word_ratio',f'{feature}_length_ratio', f'{feature}_stopword_ratio',
                     f'{feature}_punct_ratio', "karp_tfidf_scores"]

    # ngrams finding
    for n in range(2,5):
        col = f"{feature}_{n}grams"
        dataframe[col] = dataframe.swifter.apply(lambda row: count_ngrams(row[feature], n), axis=1)

        # ngrams coocurrence count
        dataframe[f"{n}grams_cnt"] = dataframe.swifter.apply(lambda row: count_cooccurring_ngrams(row[f"text_{n}grams"], row[f"prompt_text_{n}grams"]), axis=1)

        dataframe.drop([f"text_{n}grams", f"prompt_text_{n}grams"], axis=1, inplace=True)

        normalize_cols.append(f"{n}grams_cnt")


    dataframe = dataframe.replace([np.inf, -np.inf], np.nan).fillna(0)

    for col in normalize_cols:
        dataframe = normalize_col(dataframe, col)


    return dataframe

preprocessed_df = feature_engineer(df_train)

"""**Extract feature columns**"""
print(preprocessed_df.columns.to_list)

FEATURE_COLUMNS = preprocessed_df.drop(columns = ['tokenized_text_len', 'tokenized_prompt_text_len', 'prompt_id', 'text', 'prompt_question',
                                           'prompt_title', 'prompt_text', 'content', 'wording'], axis = 1).columns.to_list()

print(FEATURE_COLUMNS)
"""# Split and oversample dataset"""

from sklearn.model_selection import train_test_split
training, test_df = train_test_split(preprocessed_df, test_size=0.2, random_state=42, stratify=preprocessed_df["prompt_id"])
train_df, valid_df= train_test_split(training, test_size=0.2, random_state=42, stratify=training["prompt_id"])


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

train_df = oversample_df(train_df)

"""## Define data loaders"""

class CommonLitDataset(Dataset):
    #TODO: aggiungere max len specifica per tutti e 3 i campi
    def __init__(self, data, maxlen, tokenizer, input_cols, feature_cols, target_cols, split=True, padding=True):
        #Store the contents of the file in a pandas dataframe
        self.df = data.reset_index()
        #Initialize the tokenizer for the desired transformer model
        self.tokenizer = tokenizer
        #Maximum length of the tokens list to keep all the sequences of fixed size
        self.maxlen = maxlen
        #list of input columns
        self.input_cols = input_cols
        #list of target columns
        self.target_cols = target_cols
        #list of feature columns
        self.feature_cols = feature_cols
        self.prompt_text=[]
        self.prompt_question=[]
        self.summary=[]
        self.split=split
        self.padding=padding


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        #Select the sentence and label at the specified index in the data frame
        prompt_text_tokens=[]
        prompt_question_tokens=[]
        summary_tokens=[]
        #["text","prompt_question","prompt_text"]

        if(self.split==True):
            prompt_text_tokens=['[CLS]'] + self.tokenizer.tokenize(self.df.loc[index, self.input_cols[2]])
            prompt_question_tokens=['[CLS]'] + self.tokenizer.tokenize(self.df.loc[index, self.input_cols[1]])
            summary_tokens=['[CLS]'] + self.tokenizer.tokenize(self.df.loc[index, self.input_cols[0]])
            if len(prompt_text_tokens) < self.maxlen:
                if self.padding:
                    prompt_text_tokens = prompt_text_tokens + ['[PAD]' for _ in range(self.maxlen - len(prompt_text_tokens))]+ ['[SEP]']
                else: prompt_text_tokens = prompt_text_tokens + ['[SEP]']
            else:
                prompt_text_tokens = prompt_text_tokens[:self.maxlen-1] + ['[SEP]']
            if len(prompt_question_tokens) < self.maxlen:
                if self.padding:
                    prompt_question_tokens = prompt_question_tokens + ['[PAD]' for _ in range(self.maxlen - len(prompt_question_tokens))]+ ['[SEP]']
                else: prompt_question_tokens = prompt_question_tokens + ['[SEP]']
            else:
                prompt_question_tokens = prompt_question_tokens[:self.maxlen-1] + ['[SEP]']
            if len(summary_tokens) < self.maxlen:
                if self.padding:
                    summary_tokens = summary_tokens + ['[PAD]' for _ in range(self.maxlen - len(summary_tokens))]+ ['[SEP]']
                else: summary_tokens = summary_tokens + ['[SEP]']
            else:
                summary_tokens = summary_tokens[:self.maxlen-1] + ['[SEP]']

            #Obtain the indices of the tokens in the BERT Vocabulary
            prompt_text_input_ids = self.tokenizer.convert_tokens_to_ids(prompt_text_tokens)
            prompt_question_tokens_input_ids = self.tokenizer.convert_tokens_to_ids(prompt_question_tokens)
            summary_tokens_input_ids = self.tokenizer.convert_tokens_to_ids(summary_tokens)

            prompt_text_input_ids = torch.tensor(prompt_text_input_ids)
            prompt_question_tokens_input_ids = torch.tensor(prompt_question_tokens_input_ids)
            summary_tokens_input_ids = torch.tensor(summary_tokens_input_ids)
            input_ids=[prompt_text_input_ids, prompt_question_tokens_input_ids, summary_tokens_input_ids]

        else:
            #Select the sentence and label at the specified index in the data frame
            tokens=[]
            for col in self.input_cols:
              temp_tokens = self.tokenizer.tokenize(self.df.loc[index, col])
              tokens = tokens + temp_tokens + ['[SEP]']

            #Preprocess the text to be suitable for the lstm
            tokens = ['[CLS]'] + tokens
            if len(tokens) < self.maxlen:
                if self.padding:
                  input_length=len(tokens)
                  tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
            # else:
            #     tokens = tokens[:self.maxlen-1] + ['[SEP]']
            #     tokens = tokens + ['[SEP]']

            #Obtain the indices of the tokens in the BERT Vocabulary
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor(input_ids)
        target=[]
        try:
            target = self.df.loc[index, self.target_cols]
        except Exception:
            pass
        try:
            feature = self.df.loc[index, self.feature_cols]
        except Exception as e:
            raise e
        target = torch.tensor(target, dtype=torch.float32)
        feature = torch.tensor(feature, dtype=torch.float32)
        input_length=len(input_ids)
        return input_ids, input_length, target, feature

from transformers import BertTokenizer

model_name = 'huggingface-bert/bert-base-cased'
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

print(" Def model")
MAX_LEN=2000
BATCH_SIZE=1
split=False
padding= False
#input cols are the one for LSTM input
# Feature col are the one for dense layer input

input_cols=["text","prompt_question","prompt_text"]
target_cols=["content", "wording"]
train_set = CommonLitDataset(data=train_df, maxlen=MAX_LEN, tokenizer=tokenizer, input_cols=input_cols, feature_cols=FEATURE_COLUMNS, target_cols=target_cols, split=split, padding=padding)
valid_set = CommonLitDataset(data=valid_df, maxlen=MAX_LEN, tokenizer=tokenizer, input_cols=input_cols, feature_cols=FEATURE_COLUMNS, target_cols=target_cols, split=split, padding=padding)
test_set = CommonLitDataset(data=test_df, maxlen=MAX_LEN, tokenizer=tokenizer, input_cols=input_cols, feature_cols=FEATURE_COLUMNS, target_cols=target_cols, split=split, padding=padding)

train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

"""## Define model and evaluation functions"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim1, hidden_dim2, n_added_feature, output_dim, n_layers,
                 bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim1,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim1 * 2+n_added_feature, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths, feature, device):
        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(embedded, text_lengths.to('cpu'), batch_first=True, enforce_sorted=False).to(device)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        rel = self.relu(cat)
        x=torch.cat((rel, feature), dim=1)
        dense1 = self.fc1(x)
        drop = self.dropout(dense1)
        preds = self.fc2(drop)
        return preds

def evaluate(model, criterion, dataloader, device):
    model.eval()
    mean_acc, mean_loss, count = 0, 0, 0

    with torch.no_grad():
        for input_ids, text_length, target, feature in (dataloader):

            input_ids, text_length, target, feature = input_ids.to(device), text_length.to(device), target.to(device), feature.to(device)
            output = model(input_ids, text_length, feature, device)

            mean_loss += criterion(output, target).item()
            #mean_loss += get_rmse(output, target.type_as(output)).item()
            count += 1

    return mean_loss/count


from tqdm import tqdm, trange

def train(model, criterion, optimizer, train_loader, val_loader, test_loader, epochs, device, e_s=True):
    tl=10
    val_loss=10
    test_loss=0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for (input_ids, input_length, target, feature) in tqdm(train_loader):
            optimizer.zero_grad()
            feature =feature.to(device)
            input_ids, target = input_ids.to(device), target.to(device)
            input_length= input_length.to(device)
            output = model(input_ids, input_length, feature, device)
            loss = criterion(output, target.type_as(output))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        temp_tl = train_loss/len(train_loader)
        if test_loader!=None:
            test_loss=evaluate(model=model, criterion = criterion, dataloader=test_loader, device=device)
        if val_loader!= None:
            temp_val_loss = evaluate(model=model, criterion = criterion, dataloader=val_loader, device=device)
            if e_s==True:
                if temp_val_loss>val_loss and temp_tl<temp_val_loss:
                    print(f"Epoch {epoch} - Training Loss: {tl} Validation Loss: {val_loss}, Test Loss: {test_loss} EARLY STOPPED")
                    return tl, val_loss, test_loss     
                else:
                    val_loss=temp_val_loss
                    tl=temp_tl
                    print(f"Epoch {epoch} - Training Loss: {tl} Validation Loss: {val_loss}, Test Loss: {test_loss}")
            else:
                tl=temp_tl
                val_loss=temp_val_loss
                print(f"Epoch {epoch} - Training Loss: {tl} Validation Loss: {val_loss}, Test Loss: {test_loss}")

    return tl, val_loss, test_loss


def predict(test_loader, model, device):
    preds = []
    for input_ids, input_length, target, feature in test_loader:
        feature =feature.to(device)
        input_ids = input_ids.to(device)
        input_length= input_length.to(device)
        output = model(input_ids, input_length, feature, device)
        preds.append(output)
        del input_ids, input_length, target, feature, output
        gc.collect()
        torch.cuda.empty_cache()
    preds = torch.concat(preds)
    return preds

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class MCRMSELoss(nn.Module):
    def __init__(self, num_scored=2):
        super().__init__()
        self.rmse = RMSELoss()
        self.num_scored = num_scored

    def forward(self, yhat, y):
        score = 0
        for i in range(self.num_scored):
            score += self.rmse(yhat[:, i], y[:, i]) / self.num_scored

        return score

"""## Define hyperparameters and train"""
# print("Train model")
# LR = 1e-3
# n_added_feature=15
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = LSTM(vocab_size=30522, embedding_dim=20, bidirectional=True,dropout=0.5,hidden_dim1=100, hidden_dim2=50, n_added_feature=n_added_feature,n_layers=2,output_dim=2)
# model.to(device)
# OPTIMIZER = optim.Adam(params=model.parameters(), lr=LR)
# EPOCHS = 5
# CRITERION = MCRMSELoss()


"""## Grid search"""

embedding_dim_list=[40]
dropout_list=[0.2]
hidden1_list=[150]
hidden2_list=[120]

#TODO AUMENTARE EPOCHE
grid_search_df = pd.DataFrame()
device = "cuda" if torch.cuda.is_available() else "cpu"
for emb in embedding_dim_list:
  for dropout in dropout_list:
    for h1_dim in hidden1_list:
      for h2_dim in hidden2_list:
        LR = 1e-3
        n_added_feature=15
        model = LSTM(vocab_size=30522, embedding_dim=emb, bidirectional=True,dropout=dropout,hidden_dim1=h1_dim, hidden_dim2=h2_dim, n_added_feature=n_added_feature,n_layers=2,output_dim=2)
        model.to(device)
        OPTIMIZER = optim.Adam(params=model.parameters(), lr=LR)
        EPOCHS = 10
        CRITERION = MCRMSELoss()
        tl, vl, test_l= train(model=model,
              criterion=CRITERION,
              optimizer=OPTIMIZER,
              train_loader=train_loader,
              val_loader=valid_loader,
              test_loader=test_loader,
              epochs = EPOCHS,
              device = device,
              e_s= False)
        hyperparams= str(emb) + ", " + str(dropout)+ ", " + str(h1_dim)+ ", " + str(h2_dim)
        new_row = {'Emb, dropout, h1, h2':hyperparams, 'Training score':tl, 'Validation':vl, 'Test':test_l}
        grid_search_df = grid_search_df.append(new_row, ignore_index=True)

grid_search_df.to_csv("grid_search.csv")