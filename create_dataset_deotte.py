#Import data
from ast import literal_eval
import pandas as pd
import tqdm
from utils import *
import os

def create_dfs(name):

    config_data = read_file_in_dir('./', name + '.json')

    train_df = pd.read_csv(config_data["dataset"]["train_csv"])

    test_names, test_texts = [], []
    for f in list(os.listdir(config_data["dataset"]["train_text"])):
        test_names.append(f.replace('.txt', ''))
        test_texts.append(open(config_data["dataset"]["test_text"] + '/' + f, 'r').read())
    test_texts = pd.DataFrame({'id': test_names, 'text': test_texts})

    test_names, train_texts = [], [] #Should this be train_names?
    for f in tqdm(list(os.listdir(config_data["dataset"]["train_text"]))):
        test_names.append(f.replace('.txt', ''))
        train_texts.append(open(config_data["dataset"]["train_text"] + '/' + f, 'r').read())
    train_text_df = pd.DataFrame({'id': test_names, 'text': train_texts})

    return test_texts, train_text_df, train_df

#Convert train-text to NER labels:
def convert_text_to_ner(name, train_text_df, train_df):

    config_data = read_file_in_dir('./', name + '.json')

    if config_data["model_load"]["LOAD_TOKENS_FROM"] == "NONE":
        
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
        token_path = config_data['model_load']["LOAD_TOKENS_FROM"]
        train_text_df = pd.read_csv(f'{token_path}/train_NER.csv')
        # pandas saves lists as string, we must convert back
        train_text_df.entities = train_text_df.entities.apply(lambda x: literal_eval(x) )

    return train_text_df


LABEL_ALL_SUBTOKENS = True


#Train and validation dataloaders
    
# CHOOSE VALIDATION INDEXES (that match my TF notebook)
IDS = train_df.id.unique()
print('There are',len(IDS),'train texts. We will split 90% 10% for validation.')

# TRAIN VALID SPLIT 90% 10%
np.random.seed(42)
train_idx = np.random.choice(np.arange(len(IDS)),int(0.9*len(IDS)),replace=False)
valid_idx = np.setdiff1d(np.arange(len(IDS)),train_idx)
np.random.seed(None)

# CREATE TRAIN SUBSET AND VALID SUBSET
data = train_text_df[['id','text', 'entities']]
train_dataset = data.loc[data['id'].isin(IDS[train_idx]),['text', 'entities']].reset_index(drop=True)
test_dataset = data.loc[data['id'].isin(IDS[valid_idx])].reset_index(drop=True)

print("FULL Dataset: {}".format(data.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))