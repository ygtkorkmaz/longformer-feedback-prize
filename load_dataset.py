from constants import *
import torch
import os
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
import numpy as np
from transformers import AutoTokenizer

from constants import *


class FeedbackDataset(Dataset):

    ## CREATE DICTIONARIES THAT WE CAN USE DURING TRAIN AND INFER - Moved them to constants.
    # output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
    #         'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']

    # labels_to_ids = {v:k for k,v in enumerate(output_labels)}
    # ids_to_labels = {k:v for k,v in enumerate(output_labels)}
    ##
    
    def __init__(self, config_data, dataframe, tokenizer, max_len, get_wids):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.get_wids = get_wids # True for validation
        self.config_data = config_data

    def __getitem__(self, index):
        # GET TEXT AND WORD LABELS 
        text = self.data.text[index]        
        # word_labels = self.data.entities[index] if not self.get_wids else None
        # TRIAL
        word_labels = self.data.entities[index]

        # TOKENIZE TEXT
        encoding = self.tokenizer(text.split(),
                                is_split_into_words=True,
                                #return_offsets_mapping=True, 
                                padding='max_length', 
                                truncation=True, 
                                max_length=self.max_len)
        word_ids = encoding.word_ids()  
        
        # CREATE TARGETS
        # if not self.get_wids:
        #     previous_word_idx = None
        #     label_ids = []
        #     for word_idx in word_ids:                            
        #         if word_idx is None:
        #             label_ids.append(-100)
        #         elif word_idx != previous_word_idx:              
        #             label_ids.append( target_id_map[word_labels[word_idx]] )
        #         else:
        #             ## We can simplify this part by always labeling all tokens
        #             if self.config_data['experiment']['LABEL_ALL_SUBTOKENS']:
        #                 label_ids.append( target_id_map[word_labels[word_idx]] )
        #             else:
        #                 label_ids.append(-100)
        #             ##
        #         previous_word_idx = word_idx
        #     encoding['labels'] = label_ids

        ## TRIAL FOR VAL
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:                            
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:              
                label_ids.append( target_id_map[word_labels[word_idx]] )
            else:
                ## We can simplify this part by always labeling all tokens
                if self.config_data['experiment']['LABEL_ALL_SUBTOKENS']:
                    label_ids.append( target_id_map[word_labels[word_idx]] )
                else:
                    label_ids.append(-100)
                ##
            previous_word_idx = word_idx
        encoding['labels'] = label_ids
            

        # CONVERT TO TORCH TENSORS
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        if self.get_wids: 
            word_ids2 = [w if w is not None else -1 for w in word_ids]
            item['wids'] = torch.as_tensor(word_ids2)
        
        return item

    def __len__(self):
        return self.len


def create_dfs(config_data):

    # config_data = read_file_in_dir('./', name + '.json')

    train_df = pd.read_csv(config_data["dataset"]["train_csv"])

    # test_names, test_texts = [], []
    # for f in list(os.listdir(config_data["dataset"]["train_text"])):
    #     test_names.append(f.replace('.txt', ''))
    #     test_texts.append(open(config_data["dataset"]["test_text"] + '/' + f, 'r').read())
    # test_texts = pd.DataFrame({'id': test_names, 'text': test_texts})

    train_names, train_texts = [], [] #Should this be train_names?
    for f in tqdm(list(os.listdir(config_data["dataset"]["train_text"]))):
        train_names.append(f.replace('.txt', ''))
        train_texts.append(open(config_data["dataset"]["train_text"] + '/' + f, 'r').read())
    train_text_df = pd.DataFrame({'id': train_names, 'text': train_texts})

    return train_text_df, train_df


def convert_text_to_ner(config_data, train_text_df, train_df):

    # config_data = read_file_in_dir('./', name + '.json')

    if config_data["experiment"]["LOAD_TOKENS_FROM"] == "NONE":
        
        all_entities = []

        for ii,i in enumerate(train_text_df.iterrows()):
            if ii%100==0: print(ii,', ',end='')
            total = i[1]['text'].split().__len__()
            entities = ["O"]*total
            for j in train_df[train_df['id'] == i[1]['id']].iterrows():
                discourse = j[1]['discourse_type']
                list_ix = [int(x) for x in j[1]['predictionstring'].split(' ')]
                entities[list_ix[0]] = f"B-{discourse}"
                for k in list_ix[1:]: entities[k] = f"I-{discourse}"
            all_entities.append(entities)
        
        train_text_df['entities'] = all_entities
        train_text_df.to_csv('train_NER.csv',index=False)
        
    else:
        token_path = config_data['experiment']["LOAD_TOKENS_FROM"]
        train_text_df = pd.read_csv(f'{token_path}/train_NER.csv')
        # pandas saves lists as string, we must convert back
        train_text_df.entities = train_text_df.entities.apply(lambda x: literal_eval(x) )

    return train_text_df


def get_dataset(config_data, tokenizer):
    train_text_df, train_df = create_dfs(config_data)
    train_text_df = convert_text_to_ner(config_data, train_text_df, train_df)
    IDS = train_df.id.unique()
    print('There are',len(IDS),'train texts. We will split 80%-10%-10% for validation and test.')

    # TRAIN VALID TEST SPLIT 80% 10% 10%
    np.random.seed(42)
    permuted_idx = np.random.permutation(len(IDS))
    train_idx = permuted_idx[:int(0.8 * len(IDS))]
    valid_idx = permuted_idx[int(0.8 * len(IDS)): int(0.9 * len(IDS))]
    test_idx = permuted_idx[int(0.9 * len(IDS)):]
    # train_idx = np.random.choice(np.arange(len(IDS)),int(0.9*len(IDS)),replace=False)
    # valid_idx = np.setdiff1d(np.arange(len(IDS)),train_idx)
    np.random.seed(None)

    # CREATE TRAIN SUBSET AND VALID SUBSET
    data = train_text_df[['id','text', 'entities']]
    train_dataset = data.loc[data['id'].isin(IDS[train_idx]),['text', 'entities']].reset_index(drop=True)
    val_dataset = data.loc[data['id'].isin(IDS[valid_idx]),['text', 'entities']].reset_index(drop=True)
    test_dataset = data.loc[data['id'].isin(IDS[test_idx]),['text', 'entities']].reset_index(drop=True)
    # val_dataset = data.loc[data['id'].isin(IDS[valid_idx])].reset_index(drop=True)
    # test_dataset = data.loc[data['id'].isin(IDS[test_idx])].reset_index(drop=True)

    # tokenizer = AutoTokenizer.from_pretrained(config_data['model']['transformer_path']) 
    training_set = FeedbackDataset(config_data, train_dataset, tokenizer, config_data['experiment']['max_length'], False)
    validation_set = FeedbackDataset(config_data, val_dataset, tokenizer, config_data['experiment']['max_length'], True)
    testing_set = FeedbackDataset(config_data, test_dataset, tokenizer, config_data['experiment']['max_length'], True)

    train_loader = DataLoader(training_set,
                              batch_size=config_data['experiment']['batch_size'],
                              shuffle=True, num_workers=config_data['experiment']['num_workers'],
                              pin_memory=True)
    val_loader = DataLoader(validation_set,
                              batch_size=config_data['experiment']['batch_size'],
                              shuffle=False, num_workers=config_data['experiment']['num_workers'],
                              pin_memory=True)
    test_loader = DataLoader(testing_set,
                              batch_size=config_data['experiment']['batch_size'],
                              shuffle=False, num_workers=config_data['experiment']['num_workers'],
                              pin_memory=True)
    
    val_dataset = data.loc[data['id'].isin(IDS[valid_idx])].reset_index(drop=True)
    test_dataset = data.loc[data['id'].isin(IDS[test_idx])].reset_index(drop=True)
    val_raw_dataset = train_df.loc[train_df['id'].isin(IDS[valid_idx])].reset_index(drop=True)
    test_raw_dataset = train_df.loc[train_df['id'].isin(IDS[test_idx])].reset_index(drop=True)
                              
    return train_loader, val_loader, test_loader, val_dataset, test_dataset, val_raw_dataset, test_raw_dataset