import numpy as np, os 
import pandas as pd, gc 
from tqdm import tqdm

from create_dataset_deotte import *
from dataloader_deotte import *
from model_deotte import *
from utils import *

from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import accuracy_score


#Setup
class Experiment(object):
    def __init__(self, name):
        
        #Change this
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = read_file_in_dir('./', name + '.json')

        #Import data and convert to NER tokens - see create_dataset_deotte
        test_texts, train_text_df, train_df = create_dfs(name)
        train_df = convert_text_to_ner(name, train_text_df, train_df)

        if config['DOWNLOADED_MODEL_PATH'] == "NONE":
            config['DOWNLOADED_MODEL_PATH'] = 'model'

        tokenizer = AutoTokenizer.from_pretrained(config['DOWNLOADED_MODEL_PATH']) 
        training_set = dataset(train_dataset, tokenizer, config['max_length'], False)
        testing_set = dataset(test_dataset, tokenizer, config['max_length'], True)

        # TRAIN DATASET AND VALID DATASET
        train_params = {'batch_size': config['train_batch_size'],
                        'shuffle': True,
                        'num_workers': 2,
                        'pin_memory':True
                        }

        test_params = {'batch_size': config['valid_batch_size'],
                        'shuffle': False,
                        'num_workers': 2,
                        'pin_memory':True
                        }

        training_loader = DataLoader(training_set, **train_params)
        testing_loader = DataLoader(testing_set, **test_params)

        # TEST DATASET
        test_texts_set = dataset(test_texts, tokenizer, config['model_config']['max_length'], True)
        test_texts_loader = DataLoader(test_texts_set, **test_params)

        
        #Creates configuration files for particular model
        Model(config)

        #Instantiates configuration for model
        config_model = AutoConfig.from_pretrained(config["model_load"]['DOWNLOADED_MODEL_PATH'] +'/config.json') 
        
        #Calls model
        model = AutoModelForTokenClassification.from_pretrained(
                        config["model_load"]['DOWNLOADED_MODEL_PATH'] + '/pytorch_model.bin',config=config_model)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config["model_config"]['learning_rates'][0])
        
        #Def train

        # https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
        
        def train_epoch(epoch):
            tr_loss, tr_accuracy = 0, 0
            nb_tr_examples, nb_tr_steps = 0, 0
            #tr_preds, tr_labels = [], []
            
            # put model in training mode
            model.train()
            
            for idx, batch in enumerate(training_loader):
                
                ids = batch['input_ids'].to(config['device'], dtype = torch.long)
                mask = batch['attention_mask'].to(config['device'], dtype = torch.long)
                labels = batch['labels'].to(config['device'], dtype = torch.long)

                loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels,
                                    return_dict=False)
                tr_loss += loss.item()

                nb_tr_steps += 1
                nb_tr_examples += labels.size(0)
                
                if idx % 200==0:
                    loss_step = tr_loss/nb_tr_steps
                    print(f"Training loss after {idx:04d} training steps: {loss_step}")
                
                # compute training accuracy
                flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
                active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
                flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
                
                # only compute accuracy at active labels
                active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
                #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
                
                labels = torch.masked_select(flattened_targets, active_accuracy)
                predictions = torch.masked_select(flattened_predictions, active_accuracy)
                
                #tr_labels.extend(labels)
                #tr_preds.extend(predictions)

                tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
                tr_accuracy += tmp_tr_accuracy
            
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(), max_norm=config['max_grad_norm']
                )
                
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss = tr_loss / nb_tr_steps
            tr_accuracy = tr_accuracy / nb_tr_steps
            print(f"Training loss epoch: {epoch_loss}")
            print(f"Training accuracy epoch: {tr_accuracy}")


        # LOOP TO TRAIN MODEL (or load model)
        def train():

            ##FIX NAMING CONVENTION HERE
            VER = config['model_load']["VER"]
            if config['model_load']["LOAD_MODEL_FROM"] == "NONE":
                
                for epoch in range(config['model_config']['epochs']):
                    print(f"### Training epoch: {epoch + 1}")
                    for g in optimizer.param_groups: 
                        g['lr'] = config['model_config']['learning_rates'][epoch]
                    lr = optimizer.param_groups[0]['lr']
                    print(f'### LR = {lr}\n')
                    
                    train_epoch(epoch)
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                torch.save(model.state_dict(), f'longformer_v{VER}.pt')
            else:
                model.load_state_dict(
                    torch.load(
                    f'{config['model_load']["LOAD_MODEL_FROM"]}/longformer_v{VER}.pt'))
                print('Model loaded.')

        #Inference
        def inference(batch):
                        
            # MOVE BATCH TO GPU AND INFER
            ids = batch["input_ids"].to(config['device'])
            mask = batch["attention_mask"].to(config['device'])
            outputs = model(ids, attention_mask=mask, return_dict=False)
            all_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy() 

            # INTERATE THROUGH EACH TEXT AND GET PRED
            predictions = []
            for k,text_preds in enumerate(all_preds):
                token_preds = [ids_to_labels[i] for i in text_preds]

                prediction = []
                word_ids = batch['wids'][k].numpy()  
                previous_word_idx = -1
                for idx,word_idx in enumerate(word_ids):                            
                    if word_idx == -1:
                        pass
                    elif word_idx != previous_word_idx:              
                        prediction.append(token_preds[idx])
                        previous_word_idx = word_idx
                predictions.append(prediction)
            
            return predictions

        # https://www.kaggle.com/zzy990106/pytorch-ner-infer
        # code has been modified from original
        def get_predictions(df=test_dataset, loader=testing_loader):
            
            # put model in training mode
            model.eval()
            
            # GET WORD LABEL PREDICTIONS
            y_pred2 = []
            for batch in loader:
                labels = inference(batch)
                y_pred2.extend(labels)

            final_preds2 = []
            for i in range(len(df)):

                idx = df.id.values[i]
                #pred = [x.replace('B-','').replace('I-','') for x in y_pred2[i]]
                pred = y_pred2[i] # Leave "B" and "I"
                preds = []
                j = 0
                while j < len(pred):
                    cls = pred[j]
                    if cls == 'O': j += 1
                    else: cls = cls.replace('B','I') # spans start with B
                    end = j + 1
                    while end < len(pred) and pred[end] == cls:
                        end += 1
                    
                    if cls != 'O' and cls != '' and end - j > 7:
                        final_preds2.append((idx, cls.replace('I-',''),
                                            ' '.join(map(str, list(range(j, end))))))
                
                    j = end
                
            oof = pd.DataFrame(final_preds2)
            oof.columns = ['id','class','predictionstring']

            return oof



        valid = train_df.loc[train_df['id'].isin(IDS[valid_idx])]

        # OOF PREDICTIONS
        oof = get_predictions(test_dataset, testing_loader)

        # COMPUTE F1 SCORE
        f1s = []
        CLASSES = oof['class'].unique()
        print()
        for c in CLASSES:
            pred_df = oof.loc[oof['class']==c].copy()
            gt_df = valid.loc[valid['discourse_type']==c].copy()
            f1 = score_feedback_comp(pred_df, gt_df)
            print(c,f1)
            f1s.append(f1)
        print()
        print('Overall',np.mean(f1s))
        print()