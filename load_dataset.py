from constants import *
import torch
import csv, os
from torch.utils.data import DataLoader
from utils import prepare_training_data, prepare_test_data
import pandas as pd

class FeedbackDataset:
    def __init__(self, samples, max_len, tokenizer):
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.length = len(samples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        input_labels = self.samples[idx]["input_labels"]
        input_labels = [target_id_map[x] for x in input_labels]
        other_label_id = target_id_map["O"]

        # add start token id to the input_ids
        input_ids = [self.tokenizer.cls_token_id] + input_ids
        input_labels = [other_label_id] + input_labels

        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[: self.max_len - 1]
            input_labels = input_labels[: self.max_len - 1]        

        # add end token id to the input_ids
        input_ids = input_ids + [self.tokenizer.sep_token_id]
        input_labels = input_labels + [other_label_id]

        attention_mask = [1] * len(input_ids)

        ids = torch.tensor(input_ids, dtype=torch.long)
        mask = torch.tensor(attention_mask, dtype=torch.long)
        targets = torch.tensor(input_labels, dtype=torch.long)

        return ids, mask, targets
        # return {
        #     "ids": torch.tensor(input_ids, dtype=torch.long),
        #     "mask": torch.tensor(attention_mask, dtype=torch.long),
        #     "targets": torch.tensor(input_labels, dtype=torch.long),
        # }


class FeedbackDatasetV2:
    def __init__(self, samples, max_len, tokenizer):
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.length = len(samples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        input_labels = self.samples[idx]["input_labels"]
        input_labels = [target_id_map[x] for x in input_labels]
        other_label_id = target_id_map["O"]
        padding_label_id = target_id_map["PAD"]
        # print(input_ids)
        # print(input_labels)

        # add start token id to the input_ids
        input_ids = [self.tokenizer.cls_token_id] + input_ids
        input_labels = [other_label_id] + input_labels

        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[: self.max_len - 1]
            input_labels = input_labels[: self.max_len - 1]

        # add end token id to the input_ids
        input_ids = input_ids + [self.tokenizer.sep_token_id]
        input_labels = input_labels + [other_label_id]

        attention_mask = [1] * len(input_ids)

        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            if self.tokenizer.padding_side == "right":
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                input_labels = input_labels + [padding_label_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
            else:
                input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
                input_labels = [padding_label_id] * padding_length + input_labels
                attention_mask = [0] * padding_length + attention_mask

        ids =  torch.tensor(input_ids, dtype=torch.long)
        mask = torch.tensor(attention_mask, dtype=torch.long)
        targets = torch.tensor(input_labels, dtype=torch.long)
        return ids, mask, targets
        # return {
        #     "ids": torch.tensor(input_ids, dtype=torch.long),
        #     "mask": torch.tensor(attention_mask, dtype=torch.long),
        #     "targets": torch.tensor(input_labels, dtype=torch.long),
        # }


class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output = dict()
        output["ids"] = [sample["ids"] for sample in batch]
        output["mask"] = [sample["mask"] for sample in batch]
        output["targets"] = [sample["targets"] for sample in batch]
        padding_label_id = target_id_map["PAD"]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["ids"]]
            output["mask"] = [s + (batch_max - len(s)) * [0] for s in output["mask"]]
            output["targets"] = [s + (batch_max - len(s)) * [padding_label_id] for s in output["targets"]]
        else:
            output["ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["ids"]]
            output["mask"] = [(batch_max - len(s)) * [0] + s for s in output["mask"]]
            output["targets"] = [(batch_max - len(s)) * [padding_label_id] + s for s in output["targets"]]

        # convert to tensors
        output["ids"] = torch.tensor(output["ids"], dtype=torch.long)
        output["mask"] = torch.tensor(output["mask"], dtype=torch.long)
        output["targets"] = torch.tensor(output["targets"], dtype=torch.long)

        ids = output["ids"]
        mask = output["mask"]
        targets = output["targets"]

        return ids, mask, targets


class FeedbackDatasetTest:
    def __init__(self, samples, max_len, tokenizer):
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.length = len(samples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        input_ids = [self.tokenizer.cls_token_id] + input_ids

        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[: self.max_len - 1]

        # add end token id to the input_ids
        input_ids = input_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        return {
            "ids": input_ids,
            "mask": attention_mask,
        }


class CollateTest:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output = dict()
        output["ids"] = [sample["ids"] for sample in batch]
        output["mask"] = [sample["mask"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["ids"]]
            output["mask"] = [s + (batch_max - len(s)) * [0] for s in output["mask"]]
        else:
            output["ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["ids"]]
            output["mask"] = [(batch_max - len(s)) * [0] + s for s in output["mask"]]

        # convert to tensors
        output["ids"] = torch.tensor(output["ids"], dtype=torch.long)
        output["mask"] = torch.tensor(output["mask"], dtype=torch.long)

        return output
    
def get_dataset(experiment_config, tokenizer):
    NUM_JOBS = 12
    max_length = experiment_config['dataset']['max_length']
    df = pd.read_csv(os.path.join(experiment_config['dataset']['input'], "train_folds.csv"))
    train_df = df[df["kfold"] != 4].reset_index(drop=True)
    valid_df = df[df["kfold"] == 4].reset_index(drop=True)
    # collate = Collate(tokenizer=tokenizer)

    training_samples = prepare_training_data(train_df, tokenizer, experiment_config['dataset']['input'], num_jobs=NUM_JOBS)
    valid_samples = prepare_training_data(valid_df, tokenizer, experiment_config['dataset']['input'], num_jobs=NUM_JOBS)

    # train_dataset = FeedbackDataset(training_samples, max_length, tokenizer)   
    # val_dataset = FeedbackDataset(valid_samples, max_length, tokenizer) 
    train_dataset = FeedbackDatasetV2(training_samples, max_length, tokenizer)   
    val_dataset = FeedbackDatasetV2(valid_samples, max_length, tokenizer) 
    train_loader = DataLoader(dataset=train_dataset,
                             batch_size=experiment_config['dataset']['batch_size'],
                             shuffle=True,
                             num_workers=experiment_config['dataset']['num_workers'],
                             #collate_fn=collate,
                             pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset,
                             batch_size=experiment_config['dataset']['batch_size'],
                             shuffle=True,
                             num_workers=experiment_config['dataset']['num_workers'],
                             #collate_fn=collate,
                             pin_memory=True)
    return train_loader, val_loader


def get_test_set(experiment_config, tokenizer):
    max_length = experiment_config['dataset']['max_length']
    df = pd.read_csv(os.path.join(experiment_config['dataset']['input'], "sample_submission.csv"))
    collate = CollateTest(tokenizer=tokenizer)

    test_samples = prepare_test_data(df, tokenizer, experiment_config['dataset']['input'])

    test_dataset = FeedbackDatasetTest(test_samples, max_length, tokenizer)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=experiment_config['dataset']['batch_size'],
                             shuffle=True,
                             num_workers=experiment_config['dataset']['num_workers'],
                             collate_fn=collate,
                             pin_memory=True)
    return test_loader
