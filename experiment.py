from cmath import inf
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

from model import FeedbackModel
from load_dataset import *
from utils import *
from constants import *
from transformers import AutoTokenizer


class Experiment(object):
    def __init__(self, name):
        experiment_config = read_file_in_dir('./', name + '.json')
        if experiment_config is None:
            raise Exception("Configuration file doesn't exist: ", name)
        self.__name = experiment_config['experiment_name']
        self.__transformer_name = experiment_config['model']['transformer_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)
        self.__tokenizer = AutoTokenizer.from_pretrained(self.__transformer_name)


        self.__train_loader, self.__val_loader = get_dataset(experiment_config, self.__tokenizer)
        self.__test_loader = get_test_set(experiment_config, self.__tokenizer)

        # self.__generation_config = experiment_config['generation']
        self.__epochs = experiment_config['experiment']['num_epochs']
        self.__learning_rate = experiment_config['experiment']['learning_rate']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.
        self.__best_loss = inf

        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__model = FeedbackModel(experiment_config['model']['transformer_name'], experiment_config['model']['num_classes'])
        params = list(self.__model.parameters())
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.AdamW(params, lr=self.__learning_rate, weight_decay=0.0001)
        self.__init_model()

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

    def __train(self):
        self.__model.train()
        training_loss = []

        for i, (ids, masks, targets) in enumerate(self.__train_loader):
            self.__optimizer.zero_grad()
            ids = ids.to(self.__device)
            masks = masks.to(self.__device)
            # targets = nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True).data
            outputs = self.__model(ids, masks)
            loss = self.__criterion(outputs, targets)
            training_loss.append(loss.item())
            loss.backward()
            self.__optimizer.step()

        return np.mean(training_loss)

    def __val(self):
        self.__model.eval()
        val_loss = 0
        loss_list = []

        with torch.no_grad():
            for i, (ids,masks, targets) in enumerate(self.__val_loader):
                ids = ids.to(self.__device)
                masks = masks.to(self.__device)
                outputs = self.__model(ids, masks)
                # targets = nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True).data
                loss = self.__criterion(outputs, targets)
                loss_list.append(loss.item())
            val_loss = np.mean(loss_list)
            if val_loss < self.__best_loss:
                self.__best_loss = val_loss
                self.__best_model = self.__model.state_dict()
                self.__save_model(model_path='best_model.pt')
                result_str = "Best Validation Loss: {}, Epoch: {}".format(self.__best_loss,
                                                                            self.__current_epoch)
                self.__log(result_str)

        return val_loss

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
        acc_list = []       
        with torch.no_grad():
            for i, (ids, masks, targets) in enumerate(self.__test_loader):
                ids = ids.to(self.__device)
                masks = masks.to(self.__device)
                outputs = self.__model(ids, masks)
                # targets = nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True).data
                loss = self.__criterion(outputs, targets)
                loss_list.append(loss.item())
                predictions = torch.argmax(outputs, axis=-1)
                # batch_acc = (predictions==targets).sum()/len(targets)
                # acc_list.append(batch_acc)
                # if visualize:
                #     self.test_visualizer(predictions, targets)
            test_loss = np.mean(loss_list)
            # accuracy = np.mean(acc_list)

        result_str = "Test Performance: Loss: {}, accuracy: {}".format(test_loss)
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