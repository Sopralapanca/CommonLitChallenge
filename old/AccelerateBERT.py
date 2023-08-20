from old.utils import preprocessText, CustomTrainer, CommonLitDataset, RegressorModel
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import TrainingArguments

summaries_train_path = "../data/summaries_train.csv"
prompt_train_path = "../data/prompts_train.csv"

summaries_test_path = "../data/summaries_test.csv"
prompt_test_path = "../data/prompts_test.csv"

train_data = pd.read_csv(summaries_train_path, sep=',', index_col=0)
prompt_data = pd.read_csv(prompt_train_path, sep=',', index_col=0)

training_data = train_data.merge(prompt_data, on='prompt_id')

string_columns = ["text", "prompt_question", "prompt_title", "prompt_text"]

for col in string_columns:
    # apply preprocessText function to each text column in the dfTrain dataframe
    training_data[col] = training_data[col].apply(lambda x: preprocessText(x))

train, test = train_test_split(training_data, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

config = {
    'model': 'bert-base-cased',
    'dropout': 0.5,
    'max_length': 512,
    'batch_size': 8,  # anything more results in CUDA OOM [for unfreezed encoder] on Kaggle GPU
    'epochs': 7,
    'lr': 3e-4,
    'enable_scheduler': True,
    'scheduler': 'CosineAnnealingWarmRestarts',
    'gradient_accumulation_steps': 2,
    'adam_eps': 1e-6,  # 1e-8 default
    'freeze_encoder': True
}

target_cols = ["content", "wording"]

tokenizer = AutoTokenizer.from_pretrained(config['model'])

train_encodings = tokenizer(train["text"].values.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val["text"].values.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test["text"].values.tolist(), truncation=True, padding=True)

target_cols_train = train[target_cols].values
target_cols_valid = val[target_cols].values
target_cols_test = test[target_cols].values

train_set = CommonLitDataset(encodings=train_encodings, labels=target_cols_train)
valid_set = CommonLitDataset(encodings=val_encodings, labels=target_cols_valid)
test_set = CommonLitDataset(encodings=test_encodings, labels=target_cols_test)

#train_loader = DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
#valid_loader = DataLoader(dataset=valid_set, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

model = RegressorModel(config).to("cuda")
default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 5,
    "log_level": "error",
    "report_to": "none",
    "full_determinism": False
}
training_args = TrainingArguments(per_device_train_batch_size=1, gradient_accumulation_steps=4, **default_args)
trainer = CustomTrainer(model=model, args=training_args,
                        train_dataset=train_set, eval_dataset=valid_set)
result = trainer.train()
print(result)
