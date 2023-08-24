import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import KFold
from tf_utils.config import CONFIG

def tf_pipeline(model, keys, n_folds=5, MAX_LEN=1024):
    def data_loader():
        prompt_data = pd.read_csv("data/prompts_train.csv", sep=',', index_col=0)
        pd_dataset = pd.read_csv("data/summaries_train.csv", sep=',', index_col=0)
        pd_dataset = pd_dataset.merge(prompt_data, on='prompt_id')
        pd_dataset['labels'] = pd_dataset.apply(lambda row: (row['content'], row['wording']), axis=1)
        return pd_dataset

    def create_data(prompt_question, summary, fold_indicators, score, model, MAX_LEN):
        folds=[]
        for _ in range(CONFIG['folds']):
            folds.append({'labels': [],
                    'input_ids': [],
                    'attention_mask':[],
                    'token_type_ids': []})
        tokenizer = AutoTokenizer.from_pretrained(model)
        tok_txt = tokenizer.batch_encode_plus(
            [(k[0] + " [SEP] " + k[1]) for k in zip(prompt_question,summary)],
                                        max_length = MAX_LEN,
                                        padding='max_length',
                                        truncation=True)
        for i in range(len(prompt_question)):
            f = int(fold_indicators[i])
            folds[f]['labels'].append(score[i])
            folds[f]['input_ids'].append(tok_txt['input_ids'][i])
            folds[f]['token_type_ids'].append(tok_txt['token_type_ids'][i])
            folds[f]['attention_mask'].append(tok_txt['attention_mask'][i])
        return folds
    
    pd_dataset = data_loader()
    kf = KFold(n_splits=n_folds, random_state=None, shuffle=True)
    for i, (_, index) in enumerate(kf.split(X=pd_dataset)):
        pd_dataset.loc[index,'fold'] = i
    data = create_data(pd_dataset[keys[0]].tolist(),
                        pd_dataset[keys[1]].tolist(),
                        pd_dataset['fold'].tolist(),
                        pd_dataset['labels'].tolist(),
                        model, MAX_LEN)
    return data
