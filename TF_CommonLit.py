import numpy as np # linear algebra
import tensorflow as tf
import pandas as pd
from scipy.stats import pearsonr
from tf_utils.prepare_dataset import tf_deBERTa_pipeline
from tf_utils.config import CONFIG
from tf_utils.utils import build_BERT_model
from utils.spell_checking import spell_correction
from sklearn.model_selection import KFold


def scheduler(epoch):
    learning_rate = CONFIG['lr']
    if epoch == 0:
        return learning_rate * 0.05
    else:
        return learning_rate * (0.6**epoch)
    
class PearsonCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
    #    pass
        #print(dir(self.model))
        self.X_val, self.Y_val = val_data
    #def on_epoch_start(self,epoch):
    #    print(f"Learning rate: {self.model.optimize.learning_rate}")
    def on_epoch_end(self, epoch, logs):
        X_val_preds = self.model.predict(self.X_val)
        #print(X_val_preds.shape,self.Y_val.shape)
        pearson_corr = pearsonr(X_val_preds.ravel(), self.Y_val)
        print("pearsonr_val (from log) =", pearson_corr[0])
        logs["val_pearsonr"] = pearson_corr[0]

# data = tf_deBERTa_pipeline(CONFIG['model'], CONFIG['keys'], CONFIG['folds'], CONFIG['max_length'])



callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

callback_es = tf.keras.callbacks.EarlyStopping(monitor='val_pearsonr',
                                               patience=2,
                                               mode='max',
                                               verbose=1,
                                               restore_best_weights=True)

callback_save = tf.keras.callbacks.ModelCheckpoint(
        'berta_CL.h5', monitor='val_pearsonr',
        verbose=1, save_best_only=True,
        save_weights_only=True, mode='max',
        save_freq='epoch')

pd_dataset = pd.read_csv("data/corrected_data.csv", sep=',', index_col=0)
kf = KFold(n_splits=CONFIG['folds'], random_state=None, shuffle=True)
for i, (_, index) in enumerate(kf.split(X=pd_dataset)):
    pd_dataset.loc[index,'fold'] = i

for fold in range(CONFIG['folds']):
    results = []
    train_data = pd_dataset.loc[pd_dataset['fold'] != fold]
    train_data_, train_labels = train_data['corrected_text'].tolist(), train_data['wording'].tolist()
    val_data = pd_dataset.loc[pd_dataset['fold'] == fold]
    val_data_, val_labels  = val_data['corrected_text'].tolist(), val_data['wording'].tolist()

    model = build_BERT_model()
    model.fit(train_data_, train_labels,
            epochs = CONFIG['epochs'],
            shuffle=True,
            callbacks = [callback_lr,
                        PearsonCallback((val_data_, val_labels)),
                        callback_es,
                        callback_save,
                        ],
            batch_size = CONFIG['batch_size'],
            validation_data= (val_data_, val_labels)
        )
    results[fold] = model.evaluate((val_data_, val_labels),
                                   batch_size = CONFIG['batch_size'])
print(f"The average result for the folds is {tf.reduce_mean(results)}")

# for fold in range(CONFIG['folds']):
#     results = []
#     train_data = [x for i,x in enumerate(data) if i!=fold]
#     val_data = data[fold]
#     train_data_,train_data_labels =(np.asarray([x['input_ids'] for x in train_data]),
#                                     np.asarray([x['attention_mask'] for x in train_data])), np.asarray([x['labels'] for x in train_data])
#     train_data_ = np.asarray(train_data_).reshape(2, -1, 1024)
#     train_data_labels = np.asarray(train_data_labels).reshape(-1, 2)
#     val_data_, val_data_labels  =  (np.asarray(val_data['input_ids']),
#                                     np.asarray(val_data['attention_mask'])), np.asarray(val_data['labels'])

#     model = build_BERT_model()
#     model.fit((train_data_[0], train_data_[1]), train_data_labels,
#             epochs = CONFIG['epochs'],
#             shuffle=True,
#             callbacks = [callback_lr,
#                         PearsonCallback((val_data_, val_data_labels)),
#                         callback_es,
#                         callback_save,
#                         ],
#             batch_size = CONFIG['batch_size'],
#             validation_data= (val_data_, val_data_labels)
#         )
#     results[fold] = model.evaluate((val_data_, val_data_labels),
#                                    batch_size = CONFIG['batch_size'])
# print(f"The average result for the folds is {tf.reduce_mean(results)}")
