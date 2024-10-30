import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
import torch
import os
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pickle
import numpy as np
import argparse
from model import load_weights
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


def initialize_model_and_components(args):
    model = Model(args)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    solver = Solver(model, model.tokenizer, criterion, optimizer, args)
    return solver


class Solver:

    def __init__(self, model, tokenizer, criterion, optimizer, args):

        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.criterion = criterion
        self.optimizer = optimizer

        self.test_loss = np.inf
        self.train_loss = np.inf

        if self.args.load_model != 'None':
            self.model = load_weights(self.model, args.load_model)

    def train(self, dl=None, lr=None, early_stopping=False, dl_test=None):

        if lr is not None:
            self.optimizer.param_groups[0]['lr'] = lr

        os.makedirs(self.args.save_training_dir, exist_ok=True)

        self.model.train()
        scaler = torch.cuda.amp.GradScaler()

        logits_all = []
        labels_all = []
        loss_list = []

        best_f1 = -np.inf
        early_stopping_counter = 0

        for epoch in range(0, self.args.epochs):

            for it, (inputs, word_ids, labels) in enumerate(tqdm(dl)):
                # clear previous gradients
                self.optimizer.zero_grad()

                # move to device
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].bool().to(self.device)
                labels = labels.to(self.device)

                # Compute the loss
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    # make predictions and get only the relevant logits
                    logits = self.model(input_ids, attention_mask)

                    # get only relevant datapoints
                    logits = logits[attention_mask]
                    labels = labels[attention_mask]

                    # calculate two losses for punctuation and capitalization
                    loss_cap = self.criterion(logits[:, :2], labels[:, 0])
                    loss_punc = self.criterion(logits[:, 2:], labels[:, 1])

                    loss = loss_punc + loss_cap  # train to predict capitalization and punctuation
                    # loss = loss_punc # train to only predict punctuation
                    # loss = loss_cap #  train to only predict capitalization

                # Backward pass and optimization
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                # move tensors to cpu
                labels = labels.detach().to('cpu')
                logits = logits.detach().to('cpu')

                # store labels and predictions
                loss_list.append(loss.item())
                labels_all.append(labels)
                logits_all.append(logits)

                # log results with tensorflow
                if it % self.args.log_steps == 0:
                    # compute the results
                    self.print_loss(loss_list, it)

                    labels_all = torch.cat(labels_all)
                    logits_all = torch.cat(logits_all)
                    preds = torch.stack((logits_all[:, :2].argmax(dim=1), logits_all[:, 2:].argmax(dim=1))).T
                    calculate_metrics(labels_all, preds, self.args.classes, dl.dataset.mapping)

                    # reset labels and preds
                    labels_all = []
                    logits_all = []

                    # save the model and loss list
                    # self.save_loss(loss_list)
                    # self.save_model(suffix=(epoch+1))

            # save final model
            self.train_loss = np.mean(loss_list)
            self.save_loss(loss_list)

            # Free up memory
            torch.cuda.empty_cache()

            # early stopping
            if (early_stopping is True) and (dl_test is not None):
                test_results = self.test(dl_test)
                f1 = test_results.loc['mean', 'f1_score']
                print(f'f1 score test: {f1}. Best f1 score: {best_f1}')
                if best_f1 < f1:
                    self.save_model(suffix='best')
                    print('new best model saved')
                    best_f1 = f1
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                if early_stopping_counter >= 5:
                    print('early stopping')
                    break
            else:
                self.save_model(suffix=(epoch + 1))

            # Free up memory
            torch.cuda.empty_cache()

    def test(self, dl=None):
        # default is to use the validation dataloader

        with torch.no_grad():
            self.model.eval()
            word_level_preds_all = []
            word_level_labels_all = []

            for it, (inputs, word_ids, labels) in enumerate(tqdm(dl, 'make predictions')):
                # prepare input
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].bool().to(self.device)
                labels = labels.to(self.device)

                # make predictions
                logits = self.model(input_ids, attention_mask)

                # get only the relevant logits
                attention_mask = ~torch.isnan(
                    word_ids)  # this is an attention mask which also sets special tokens to False

                word_ids = word_ids[attention_mask].detach()
                logits = logits[attention_mask].detach().to('cpu')
                labels = labels[attention_mask].detach().to('cpu')

                preds = torch.stack((logits[:, :2].argmax(dim=1), logits[:, 2:].argmax(dim=1))).T

                # map back to word_level_labels
                unique_words, counts = torch.unique_consecutive(word_ids,
                                                                return_counts=True)  # Find unique words and their counts in the word_ids
                word_starts = torch.cat((torch.tensor([0]), counts.cumsum(dim=0)[
                                                            :-1]))  # Calculate the start index of each word in the sorted array

                # getting back the word level labels
                # word_level_labels = torch.stack([labels[start:start+count].max(dim=0).values for start, count in zip(word_starts, counts)])
                word_level_labels = torch.tensor(
                    [(labels[start, 0].item(), labels[start + count - 1, 1].item()) for start, count in
                     zip(word_starts, counts)])

                # taking the max. this is not ideal
                # word_level_preds = torch.stack([preds[start:start+count].max(dim=0).values for start, count in zip(word_starts, counts)]) # Compute max predictions for each word

                # for the capitalization take the prediction of the first token which belongs to a word
                # for the punctuation take the prediction of the last token which belongs to the word
                word_level_preds = torch.tensor(
                    [(preds[start, 0].item(), preds[start + count - 1, 1].item()) for start, count in
                     zip(word_starts, counts)])

                # save word level prediction and labels
                word_level_preds_all.append(word_level_preds)
                word_level_labels_all.append(word_level_labels)

        word_level_preds_all = torch.vstack(word_level_preds_all)
        word_level_labels_all = torch.vstack(word_level_labels_all)

        # Free up memory
        torch.cuda.empty_cache()

        return calculate_metrics(word_level_labels_all, word_level_preds_all, self.args.classes, dl.dataset.mapping,
                                 False)

    def find_lr(self, dl, init_lr=1e-5, final_lr=0.1, n_steps=100, show_plot=True):

        # store the initial weights
        initial_model_weights, initial_optimizer_state = self.save_state()

        losses = []
        lrs = []
        lr_step = (final_lr / init_lr) ** (1 / (n_steps - 1))
        current_lr = init_lr
        smoothing_window = int(n_steps / 10)

        scaler = torch.cuda.amp.GradScaler()
        self.model.train()
        self.optimizer.param_groups[0]['lr'] = init_lr

        for step in tqdm(range(n_steps)):
            # clear previous gradients
            self.optimizer.zero_grad()

            inputs, word_ids, labels = next(iter(dl))

            # move to device
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].bool().to(self.device)
            labels = labels.to(self.device)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # make predictions and get only the relevant logits
                logits = self.model(input_ids, attention_mask)

                # get only relevant datapoints
                logits = logits[attention_mask]
                labels = labels[attention_mask]

                # calculate two losses for punctuation and capitalization
                loss_cap = self.criterion(logits[:, 0:2], labels[:, 0])
                loss_punc = self.criterion(logits[:, 2:], labels[:, 1])

                loss = loss_cap + loss_punc

            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            lrs.append(current_lr)
            losses.append(loss.item())

            # update learning rate
            current_lr *= lr_step
            self.optimizer.param_groups[0]['lr'] = current_lr
            # torch.cuda.empty_cache()

        results = pd.DataFrame({'loss': losses}, index=lrs)
        results.index.name = 'lrs'
        results = results.groupby(results.index).mean()

        smoothed_results = results.rolling(window=smoothing_window, center=True).mean()
        min_lr = smoothed_results['loss'].idxmin()
        min_loss = smoothed_results['loss'].min()

        if show_plot is True:
            # plt.plot(results.index, results['loss'], label='Loss')
            plt.plot(smoothed_results.index, smoothed_results['loss'], label='Loss_smoothed')

            plt.scatter(min_lr, min_loss, color='red', label='Min Loss', s=100)
            plt.axvline(min_lr / 10, color='red', label='Min Loss LR / 10')

            plt.xscale("log")
            plt.xlabel("Learning Rate (log scale)")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

        # restore the original weights
        self.restore_weights(initial_model_weights, initial_optimizer_state)

        # Free up memory
        torch.cuda.empty_cache()

        return min_lr / 10

    def print_loss(self, loss_list, it):
        # get the average training loss
        mean = np.mean(loss_list[-self.args.log_steps:])
        print(f'Iteration {it}: average loss: {mean:.4f}')

    def save_loss(self, loss_list):
        # save the model and loss list
        file_path = os.path.join(self.args.save_training_dir, f'{self.args.model_save_name}_loss.csv')
        pd.DataFrame({'Loss': loss_list}).to_csv(file_path)

    def save_model(self, suffix=''):
        if self.args.model_save_name != 'None':
            file_path = os.path.join(self.args.save_training_dir, f'{self.args.model_save_name}_{suffix}_last.pth')
            torch.save(self.model.state_dict(), file_path)

    def save_state(self):
        # Save the initial weights and optimizer state
        weights = {k: v.clone() for k, v in self.model.state_dict().items()}
        optimizer_state = {k: v for k, v in self.optimizer.state_dict().items()}
        return weights, optimizer_state

    def restore_weights(self, initial_model_weights, initial_optimizer_state):
        # Restore the initial weights of the model
        self.model.load_state_dict(initial_model_weights)

        # Reinitialize the optimizer with the original parameters
        new_optimizer = type(self.optimizer)(self.model.parameters(), **self.optimizer.defaults)
        new_optimizer.load_state_dict(initial_optimizer_state)
        self.optimizer = new_optimizer


def calculate_metrics(labels, preds, classes, reverse_mapping_punc, print_results=True):
    """
    calculates recall when labels and classes are in the same shape
    """

    punc_classes = range(len(classes) + 1)
    cap_classes = [0, 1]
    mapping = {0: 'O', 1: 'PERIOD', 2: 'COMMA', 3: 'COLON', 4: 'EXCLAMATION', 5: 'QUESTION'}
    reverse_mapping_punc = {item: key for key, item in mapping.items()}

    # Metrics for punctuation
    punc_prec, punc_rec, punc_f1 = precision_recall_f1score(labels[:, 1], preds[:, 1], punc_classes)
    punc_df = pd.DataFrame({'precision': punc_prec, 'recall': punc_rec, 'f1_score': punc_f1},
                           index=reverse_mapping_punc.keys()).iloc[1:]

    # Metrics for capitalization
    cap_prec, cap_rec, cap_f1 = precision_recall_f1score(labels[:, 0], preds[:, 0], cap_classes)
    cap_df = pd.DataFrame({'precision': cap_prec, 'recall': cap_rec, 'f1_score': cap_f1},
                          index=['LOWERCASE', 'UPPERCASE']).iloc[1:]

    # Combine into a single DataFrame
    results = pd.concat([cap_df, punc_df])
    results.loc['mean', :] = results.mean()

    if print_results:
        print(results)

    return results


def precision_recall_f1score(targets, preds, classes):
    precision = precision_score(targets, preds, labels=classes, average=None, zero_division=0)
    recall = recall_score(targets, preds, labels=classes, average=None, zero_division=0)
    f1 = f1_score(targets, preds, labels=classes, average=None, zero_division=0)

    return precision, recall, f1


def define_parser():
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size_train', type=int, default=16)
    parser.add_argument('--batch_size_test', type=int, default=16)
    parser.add_argument('--classes', type=str, default='.,:!?')
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--save_training_dir', type=str, default='')
    parser.add_argument('--model_save_name', type=str, default='')
    parser.add_argument('--load_model', type=str, default='', help='path to a model checkpoint')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='specify the model, can either be a Bert, Roberta, Electra or Deberta-v2 model')

    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_size', type=int, default=1000, help="Number of chunks in the training set")
    parser.add_argument('--valid_size', type=int, default=1000, help="Number of chunks in the validation set")
    parser.add_argument('--test_size', type=int, default=1000, help="Number of chunks in the test set")
    parser.add_argument('--save_data_dir', type=str, default='directory where the train, test , val split is saved')
    parser.add_argument('--books_dir', type=str, default='directory to the .txt files of the books')

    return parser
