from transformers import *
import os

# VERSION FOR SAVING MODEL WEIGHTS
VER=26

# IF VARIABLE IS NONE, THEN NOTEBOOK COMPUTES TOKENS
# OTHERWISE NOTEBOOK LOADS TOKENS FROM PATH
#LOAD_TOKENS_FROM = '../input/py-bigbird-v26'
LOAD_TOKENS_FROM = None

# IF VARIABLE IS NONE, THEN NOTEBOOK TRAINS A NEW MODEL
# OTHERWISE IT LOADS YOUR PREVIOUSLY TRAINED MODEL
#LOAD_MODEL_FROM = '../input/py-bigbird-v26'
LOAD_MODEL_FROM = None

# IF FOLLOWING IS NONE, THEN NOTEBOOK 
# USES INTERNET AND DOWNLOADS HUGGINGFACE 
# CONFIG, TOKENIZER, AND MODEL
#DOWNLOADED_MODEL_PATH = '../input/py-bigbird-v26' 
DOWNLOAD_MODEL_PATH = None

if DOWNLOADED_MODEL_PATH is None:
    DOWNLOADED_MODEL_PATH = 'model'    

MODEL_NAME = 'allenai/longformer_base_4096'

#Config

from torch import cuda
config = {'model_name': MODEL_NAME,   
         'max_length': 1024,
         'train_batch_size':4,
         'valid_batch_size':4,
         'epochs':5,
         'learning_rates': [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7],
         'max_grad_norm':10,
         'device': 'cuda' if cuda.is_available() else 'cpu'}

if DOWNLOADED_MODEL_PATH == 'model':
    os.mkdir('model')
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    tokenizer.save_pretrained('model')

    config_model = AutoConfig.from_pretrained(MODEL_NAME) 
    config_model.num_labels = 15
    config_model.save_pretrained('model')

    backbone = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, 
                                                               config=config_model)
    backbone.save_pretrained('model')