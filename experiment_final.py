from cmath import inf
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import gc

from model_final import *
from load_dataset_final import *
from utils import *
from constants import *
from sklearn.metrics import accuracy_score

class Experiment(object):
    def __init__(self, name):
        experiment_config = read_file_in_dir('./', name + '.json')
        if experiment_config is None:
            raise Exception("Configuration file doesn't exist: ", name)
        self.__name = experiment_config['experiment_name']
        self.__experiment_config = experiment_config
        # self.__transformer_name = experiment_config['model']['transformer_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)
        # self.__tokenizer = AutoTokenizer.from_pretrained(self.__transformer_name)
        self.__model, self.__tokenizer, self.__config_model = feedback_model(experiment_config)



        self.__train_loader, self.__val_loader, self.__test_loader, self.__val_raw_df, self.__test_raw_df = get_dataset(experiment_config, self.__tokenizer)

        # self.__generation_config = experiment_config['generation']
        self.__num_labels = 15
        self.__epochs = experiment_config['experiment']['num_epochs']
        self.__learning_rate = experiment_config['experiment']['learning_rate']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.
        self.__best_f1 = 0
        self.__best_f1_scores = None

        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.__criterion = nn.CrossEntropyLoss()
        if isinstance(self.__learning_rate, list):
            self.__optimizer = torch.optim.Adam(params=self.__model.parameters(), lr=self.__learning_rate[0])
        else:
            self.__optimizer = torch.optim.Adam(params=self.__model.parameters(), lr=self.__learning_rate)
        self.__init_model()
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)
        os.makedirs(self.__experiment_dir)


    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            # self.__criterion = self.__criterion.cuda()

    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            print(f"### Training epoch: {epoch + 1}")
            if isinstance(self.__learning_rate, list):
                for g in self.__optimizer.param_groups: 
                    g['lr'] = self.__learning_rate[epoch]
                lr = self.__optimizer.param_groups[0]['lr']
                print(f'### LR = {lr}\n')
            
            train_loss, train_accuracy = self.__train()
            val_loss, val_predictions_df = self.__val()
            total_f1, f1_scores = self.get_f1_score(val_predictions_df, self.__val_raw_df)
            if total_f1 >= self.__best_f1:
                self.__best_f1_scores = f1_scores
                self.__best_f1 = total_f1
                self.__best_model = self.__model.state_dict()
                self.__save_model(model_path='best_model.pt')                

            # we need to implement validation and add here
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()
        
        print('Training Ended')
        print(f1_scores)
        print('average f1: {}'.format(total_f1))
        print()
        print('Best Specs')
        print(self.__best_f1_scores)
        print('Best avarage f1: {}'.format(self.__best_f1))
        print()
        
        # housekeeping
        gc.collect() 
        torch.cuda.empty_cache()

    def __train(self):
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        loss_list = []
        acc_list = []
        #tr_preds, tr_labels = [], []
        
        # put model in training mode
        self.__model.train()
        
        for idx, batch in enumerate(self.__train_loader):
            
            ids = batch['input_ids'].to(self.__device, dtype = torch.long)
            mask = batch['attention_mask'].to(self.__device, dtype = torch.long)
            labels = batch['labels'].to(self.__device, dtype = torch.long)

            loss, tr_logits = self.__model(input_ids=ids, attention_mask=mask, labels=labels,
                                return_dict=False)
            tr_loss += loss.item()
            loss_list.append(loss.item())

            nb_tr_steps += 1
            nb_tr_examples += labels.size(0)
            
            if idx % 200==0:
                loss_step = tr_loss/nb_tr_steps
                print(f"Training loss after {idx:04d} training steps: {loss_step}")
                print(f"Training loss after {idx:04d} training steps: {np.mean(loss_list)}")

            
            # compute training accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(-1, self.__model.num_labels) # shape (batch_size * seq_len, num_labels)
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
            acc_list.append(tmp_tr_accuracy)
        
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=self.__model.parameters(), max_norm=self.__experiment_config['model']['max_grad_norm']
            )
            
            # backward pass
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

        epoch_loss = tr_loss / nb_tr_steps
        train_loss = np.mean(loss_list)
        # return that loss
        tr_accuracy = tr_accuracy / nb_tr_steps
        train_accuracy = np.mean(acc_list)
        # return that acc
        print(f"Training loss epoch: {train_loss}")
        print(f"Training accuracy epoch: {train_accuracy}")
        return train_loss, train_accuracy
    
    def __val(self):
        self.__model.eval()
        val_loss = 0
        loss_list = []
        final_predictions = []

        with torch.no_grad():
            for idx, batch in enumerate(self.__val_loader):

                # MOVE BATCH TO GPU AND INFER
                ids = batch['input_ids'].to(self.__device, dtype = torch.long)
                mask = batch['attention_mask'].to(self.__device, dtype = torch.long)
                labels = batch['labels'].to(self.__device, dtype = torch.long)
                loss, outputs = self.__model(ids, attention_mask=mask, labels=labels, return_dict=False)
                loss_list.append(loss.item())
                all_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy() 

                # INTERATE THROUGH EACH TEXT AND GET PRED
                predictions = []
                for k,text_preds in enumerate(all_preds):
                    token_preds = [id_target_map[i] for i in text_preds]

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

                final_predictions.extend(predictions)

            val_loss = np.mean(loss_list)

            final_preds2 = []
            for i in range(len(self.__val_raw_df)):

                idx = self.__val_raw_df.id.values[i]
                #pred = [x.replace('B-','').replace('I-','') for x in y_pred2[i]]
                pred = final_predictions[i] # Leave "B" and "I"
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
        
        return val_loss, oof
    
    def get_f1_score(self, prediction_df, target_df):
        # COMPUTE F1 SCORE
        f1s = []
        f1_dict = dict()
        CLASSES = prediction_df['class'].unique()
        print()
        for c in CLASSES:
            pred_df = prediction_df.loc[prediction_df['class']==c].copy()
            gt_df = target_df.loc[target_df['discourse_type']==c].copy()
            f1 = score_feedback_comp(pred_df, gt_df)
            print(c,f1)
            f1s.append(f1)
            f1_dict[c] = f1
        print()
        print('Overall',np.mean(f1s))
        print()
        return np.mean(f1s), f1_dict

    def test(self, model_loc=None):
        self.__best_model = self.__model
        if model_loc is not None:
            best_checkpoint = torch.load(model_loc)
        else:
            best_checkpoint = torch.load(os.path.join(self.__experiment_dir, 'best_model.pt'))
        self.__best_model.load_state_dict(best_checkpoint['model'])            
        self.__best_model = self.__best_model.to(self.__device)
        self.__best_model.eval()
        test_loss = 0
        loss_list = []
        final_predictions = []
        with torch.no_grad():
            for idx, batch in enumerate(self.__test_loader):

                # MOVE BATCH TO GPU AND INFER
                ids = batch['input_ids'].to(self.__device, dtype = torch.long)
                mask = batch['attention_mask'].to(self.__device, dtype = torch.long)
                labels = batch['labels'].to(self.__device, dtype = torch.long)
                loss, outputs = self.__model(ids, attention_mask=mask, labels=labels, return_dict=False)
                loss_list.append(loss.item())
                all_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy() 

                # INTERATE THROUGH EACH TEXT AND GET PRED
                predictions = []
                for k,text_preds in enumerate(all_preds):
                    token_preds = [id_target_map[i] for i in text_preds]

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
                final_predictions.extend(predictions)
            test_loss = np.mean(loss_list)

            final_preds2 = []
            for i in range(len(self.__test_raw_df)):

                idx = self.__test_raw_df.id.values[i]
                #pred = [x.replace('B-','').replace('I-','') for x in y_pred2[i]]
                pred = final_predictions[i] # Leave "B" and "I"
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

            test_total_f1, test_f1_scores = self.get_f1_score(oof, self.__test_raw_df)

        result_str = "Test Performance: Loss: {}, f1 Score: {}".format(test_loss, test_total_f1)
        self.__log(result_str)
        print("Test f1 Scores by Class:")
        print(test_f1_scores)

    def __save_model(self, model_path = 'latest_model.pt'):
        root_model_path = os.path.join(self.__experiment_dir, model_path)
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()