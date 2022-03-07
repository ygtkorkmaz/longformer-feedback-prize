from transformers import *
import torch
import os

class Model():

    def ___init__(self, config):


        if config["DOWNLOADED_MODEL_PATH"] == 'model':
            os.mkdir('model')
            
            tokenizer = AutoTokenizer.from_pretrained(config["model_config"]["MODEL_NAME"], add_prefix_space=True)
            tokenizer.save_pretrained('model')

            config_model = AutoConfig.from_pretrained(config["model_config"]["MODEL_NAME"]) 
            config_model.num_labels = 15
            config_model.save_pretrained('model')

            backbone = AutoModelForTokenClassification.from_pretrained(config["model_config"]["MODEL_NAME"], 
                                                                    config=config_model)
            backbone.save_pretrained('model')