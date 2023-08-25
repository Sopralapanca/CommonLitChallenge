import spacy
import contextualSpellCheck
import pandas as pd
from tf_utils.utils import gpu_selection
import tensorflow as tf
import time


def test():
   nlp = spacy.load('en_core_web_trf')
   contextualSpellCheck.add_to_pipe(nlp)
   multiple_sentences = ['Hi, im Chrsitian', 'I would like to meet u if you wnat.']
   for sentence in multiple_sentences:
      print(sentence)
      doc = nlp(sentence)
      if doc._.performed_spellCheck:
         errors = len(doc._.suggestions_spellCheck.keys())
         print(f'The spell checking has corrected {errors} errors, and the result is:')
         print(doc._.outcome_spellCheck) 

def mul_spell_correction(multiple_sentences: list, type='sm') -> dict:
   # print('{type} var can be [sm, trf]')

   spacy.require_gpu()
   nlp = spacy.load('en_core_web_'+type)
   contextualSpellCheck.add_to_pipe(nlp)
   results = {'errors': [],
              'sentence': [],
              'mispelled': []}
   for sentence in multiple_sentences:
      doc = nlp(sentence)
      results['errors'].append(len(doc._.suggestions_spellCheck.keys()))
      if doc._.performed_spellCheck:
         results['sentence'].append(doc._.outcome_spellCheck)
         results['mispelled'].append(list(doc._.suggestions_spellCheck.keys()))
      else:
         results['sentence'].append(sentence)
         results['mispelled'].append(False)
   return results
type='sm'
# spacy.require_gpu()
nlp = spacy.load('en_core_web_'+type)
contextualSpellCheck.add_to_pipe(nlp)

def spell_correction(sentence: str) -> dict:
   # print('{type} var can be [sm, trf]')
   doc = nlp(sentence)
   errors = len(doc._.suggestions_spellCheck.keys())
   if doc._.performed_spellCheck:
      corrected = doc._.outcome_spellCheck
      mispelled = list(doc._.suggestions_spellCheck.keys())
   else:
      corrected = sentence
      mispelled = False
   return pd.Series([corrected, errors, mispelled])

st = time.time()

tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)

# prompt_data = pd.read_csv("data/prompts_train.csv", sep=',', index_col=0)
# pd_dataset = pd.read_csv("data/summaries_train.csv", sep=',', index_col=0)
# pd_dataset = pd_dataset.merge(prompt_data, on='prompt_id')

# print(results)
# pd_dataset[['correct_text', 'errors', 'miss']]= pd_dataset['text'].apply(spell_correction)
pd_dataset = pd.read_csv('data/corrected_data.csv', sep=',',index_col=0)
print
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
pd_dataset.to_csv('data/corrected_data.csv')