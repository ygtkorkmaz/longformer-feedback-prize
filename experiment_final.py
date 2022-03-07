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
from transformers import AutoTokenizer
from sklearn import metrics

class Experiment(object):
    def __init__(self, name):
        experiment_config = read_file_in_dir('./', name + '.json')
        if experiment_config is None:
            raise Exception("Configuration file doesn't exist: ", name)
        self.__name = experiment_config['experiment_name']
        # self.__transformer_name = experiment_config['model']['transformer_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)
        # self.__tokenizer = AutoTokenizer.from_pretrained(self.__transformer_name)
        self.__model, self.__tokenizer, self.__config_model = feedback_model(experiment_config)



        self.__train_loader, self.__val_loader, self.__test_loader = get_dataset(experiment_config, self.__tokenizer)

        # self.__generation_config = experiment_config['generation']
        self.__num_labels = 15
        self.__epochs = experiment_config['experiment']['num_epochs']
        self.__learning_rate = experiment_config['experiment']['learning_rate']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.
        self.__best_loss = inf

        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__criterion = nn.CrossEntropyLoss()
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
            self.__criterion = self.__criterion.cuda()

    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()
        
        # housekeeping
        gc.collect() 
        torch.cuda.empty_cache()

    def __train(self):
        self.__model.train()
        training_loss = []

        for i, (ids, mask, targets) in enumerate(self.__train_loader):
            self.__optimizer.zero_grad()
            ids = ids.to(self.__device)
            mask = mask.to(self.__device)
            outputs = self.__model(ids, mask)
            # targets = nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True).data
            active_loss = mask.view(-1) == 1
            active_logits = outputs.view(-1, self.__num_labels)
            true_labels = targets.view(-1)
            predictions = active_logits.argmax(dim=-1)
            idxs = np.where(active_loss.cpu().numpy() == 1)[0]
            active_logits = active_logits[idxs]
            true_labels = true_labels[idxs].to(torch.long)
            true_labels = true_labels.to(self.__device)
            active_logits = active_logits.to(self.__device)
            loss = self.__criterion(active_logits, true_labels)
            training_loss.append(loss.item())
            loss.backward()
            self.__optimizer.step()

        return np.mean(training_loss)

    def __val(self):
        self.__model.eval()
        val_loss = 0
        loss_list = []
        f1_score_list = []

        with torch.no_grad():
            for i, (ids, mask, targets) in enumerate(self.__val_loader):
                ids = ids.to(self.__device)
                mask = mask.to(self.__device)
                outputs = self.__model(ids, mask)
                # targets = nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True).data
                active_loss = mask.view(-1) == 1
                active_logits = outputs.view(-1, self.__num_labels)
                true_labels = targets.view(-1)
                idxs = np.where(active_loss.cpu().numpy() == 1)[0]
                active_logits = active_logits[idxs]
                true_labels = true_labels[idxs].to(torch.long)
                true_labels = true_labels.to(self.__device)
                active_logits = active_logits.to(self.__device)
                loss = self.__criterion(active_logits, true_labels)
                loss_list.append(loss.item())
                predictions = active_logits.argmax(dim=-1).cpu().numpy()
                true_labels = true_labels.cpu().numpy()
                f1_score = metrics.f1_score(true_labels, predictions, average="macro")
                f1_score_list.append(f1_score)

            val_loss = np.mean(loss_list)
            val_f1_score = np.mean(f1_score_list)
            if val_loss < self.__best_loss:
                self.__best_loss = val_loss
                self.__best_model = self.__model.state_dict()
                self.__save_model(model_path='best_model.pt')
                result_str = "Best Validation Loss: {}, Epoch: {}".format(self.__best_loss,
                                                                            self.__current_epoch)
                self.__log(result_str)

        return val_loss, val_f1_score

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
        f1_score_list = []       
        with torch.no_grad():
            for i, (ids, mask, targets) in enumerate(self.__test_loader):
                ids = ids.to(self.__device)
                mask = mask.to(self.__device)
                outputs = self.__model(ids, mask)
                # targets = nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True).data
                active_loss = mask.view(-1) == 1
                active_logits = outputs.view(-1, self.__num_labels)
                true_labels = targets.view(-1)
                idxs = np.where(active_loss.cpu().numpy() == 1)[0]
                active_logits = active_logits[idxs]
                true_labels = true_labels[idxs].to(torch.long)
                true_labels = true_labels.to(self.__device)
                active_logits = active_logits.to(self.__device)
                loss = self.__criterion(active_logits, true_labels)
                loss_list.append(loss.item())
                predictions = active_logits.argmax(dim=-1).cpu().numpy()
                true_labels = true_labels.cpu().numpy()
                f1_score = metrics.f1_score(true_labels, predictions, average="macro")
                f1_score_list.append(f1_score)
            test_loss = np.mean(loss_list)
            test_f1_score = np.mean(f1_score_list)

        result_str = "Test Performance: Loss: {}, f1 Score: {}".format(test_loss, test_f1_score)
        self.__log(result_str)

        return test_loss, predictions

    def test_visualizer(self, predictions, targets):
        for prediction, target in zip(predictions, targets):
            target_type = id_target_map[target]
            predicted_type = id_target_map[prediction]
            print('Target Type: {}  Target id: {}'.format(target_type, target))
            print('Predicted Type: {}  Predicted id: {}'.format(predicted_type, prediction))

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