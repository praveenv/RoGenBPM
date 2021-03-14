import argparse
import numpy as np
import torch
from torch import nn, optim, autograd
import pandas as pd
from numpy import vstack, argmax
from sklearn.metrics import accuracy_score
from jellyfish._jellyfish import damerau_levenshtein_distance
import copy
from nltk.util import ngrams
pd.options.mode.chained_assignment = None  # default='warn'
import torchvision.transforms as transforms
from tqdm import tqdm
import math
import random

random.seed(20)


def suffix_prediction(model, X_test, Y_test, X_train, Y_train, generalized_data):

    l1_loss = nn.L1Loss()

    case_length = []
    current_case_length = 1
    X_test = X_train
    Y_test = Y_train

    for i in tqdm(range(1, X_test.shape[0]), desc = "finding case lengths"):
        current_case = X_test[i, :, 0]
        temp = torch.count_nonzero(current_case).detach().cpu().numpy()
        if temp == 1:
            current_case_length = list(range(current_case_length, 0, -1))
            case_length.extend(current_case_length)
            current_case_length = 1
        else:
            current_case_length += 1
    
    current_case_length = list(range(current_case_length, 0, -1))
    case_length.extend(current_case_length)
    total_dl_distance = 0.0
    total_time_mae = 0.0
    count = 0
    valid_case_count = 0
    for i in tqdm(range(0, X_test.shape[0]), desc="suffix prediction"):
        current_case = X_test[i, :, :]
        current_case = current_case.unsqueeze_(0)
        
        predicted_case = current_case

        number_to_predict = case_length[i]
        
        if number_to_predict > 5:
            continue
        valid_case_count += 1

        ground_truth_labels = Y_test[i:(i+number_to_predict), 0].detach().cpu().numpy().tolist()
        ground_truth_labels = [int(i) for i in ground_truth_labels]

        ground_truth_labels = map(str, ground_truth_labels)    
        ground_truth_labels = ''.join(ground_truth_labels)      

        ground_truth_timestamps = torch.sum(Y_test[i:(i+number_to_predict), 1])

        case_labels = []
        case_time_acc = torch.zeros(1).cuda()

        for j in range(0, number_to_predict):
            logits, time = model(predicted_case)
            preds = torch.argmax(logits, dim=1)[0].float()

            resource = predicted_case[0, -1, 2]

            if generalized_data:
                generalization = predicted_case[0, -1, 3]
                predicted_row = torch.tensor([preds, time, resource, generalization]).cuda()
            else:
                predicted_row = torch.tensor([preds, time, resource]).cuda()

            if (j+1) != number_to_predict:
                predicted_case[0, -(j+2), :] = predicted_case[0, -(j+1), :]
                predicted_case[0, -(j+1), :] = predicted_row

            preds = preds.detach().cpu().numpy().tolist()
            case_labels.append(preds)
            case_time_acc = case_time_acc + time[0]
        
        case_labels = [int(i) for i in case_labels]
        case_labels = map(str, case_labels)    
        case_labels = ''.join(case_labels)    
        
        dist = 1 - (damerau_levenshtein_distance(case_labels, ground_truth_labels) /
                         max((len(case_labels), len(ground_truth_labels))))
        total_dl_distance = total_dl_distance + dist
        total_time_mae = total_time_mae + l1_loss(case_time_acc[0], ground_truth_timestamps).detach().cpu().numpy()
    
    total_dl_distance = total_dl_distance / valid_case_count
    total_time_mae = total_time_mae / valid_case_count
    print("DL Distance")
    print(total_dl_distance)
    print("Timestamp MAE")
    print(total_time_mae)


class RNN(nn.Module):
    def __init__(self, total_size, input_size, embedding_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(total_size, embedding_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size + embedding_size - 1, hidden_size, num_layers, dropout = 0.2, batch_first=True)
        self.lstm_timestamp = nn.LSTM(input_size + embedding_size - 1, hidden_size, num_layers, dropout = 0.2, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.fc_timestamp = nn.Linear(hidden_size, 1)
    
    def forward(self, x):


        temp = x[:,:,0]
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()

        ht = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        ct = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()

        torch.nn.init.xavier_uniform_(h0)
        torch.nn.init.xavier_uniform_(c0)

        torch.nn.init.xavier_uniform_(ht)
        torch.nn.init.xavier_uniform_(ct)
        
        embeds = self.embedding(temp.long())

        embeds = torch.cat((embeds, x[:,:,1:]), dim=2)

        # Forward propagate LSTM
        out, _ = self.lstm(embeds,(h0,c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out_timestamp, _ = self.lstm_timestamp(embeds,(ht,ct))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out_timestamp = self.fc_timestamp(out_timestamp[:, -1, :])
        out_timestamp = torch.flatten(out_timestamp)
        return out, out_timestamp

def mean_accuracy(logits, y):
    preds = torch.argmax(logits, dim=1).float()
    return (preds == y).float().mean()

def pretty_print(*values):
    col_width = 20

    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)

    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))

def penalty(logits, y):
    scale = torch.tensor(1.).requires_grad_()
    loss = torch.nn.functional.cross_entropy(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad)

def time_penalty(logits, y):
    l1_loss = nn.L1Loss()
    scale = torch.tensor(1.).requires_grad_()
    loss = l1_loss(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad)

def run(data, suffix):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sequence_length = 10

    if data == 'orig':
        env1_X = pd.read_csv('../data/Helpdesk_env1_X.csv').to_numpy()
        env1_Y = pd.read_csv('../data/Helpdesk_env1_Y.csv').to_numpy()
        env2_X = pd.read_csv('../data/Helpdesk_env2_X.csv').to_numpy()
        env2_Y = pd.read_csv('../data/Helpdesk_env2_Y.csv').to_numpy()
        env3_X = pd.read_csv('../data/Helpdesk_env3_X.csv').to_numpy()
        env3_Y = pd.read_csv('../data/Helpdesk_env3_Y.csv').to_numpy()

    if data == 'gen':
        env1_X = pd.read_csv('../data/Helpdesk_gen_env1_X.csv').to_numpy()
        env1_Y = pd.read_csv('../data/Helpdesk_gen_env1_Y.csv').to_numpy()
        env2_X = pd.read_csv('../data/Helpdesk_gen_env2_X.csv').to_numpy()
        env2_Y = pd.read_csv('../data/Helpdesk_gen_env2_Y.csv').to_numpy()
        env3_X = pd.read_csv('../data/Helpdesk_gen_env3_X.csv').to_numpy()
        env3_Y = pd.read_csv('../data/Helpdesk_gen_env3_Y.csv').to_numpy()

    number_of_features = env1_X.shape[1]
    # Reshape into Number of sequences * length of each sequence (Ngram) * Number of features for each tuple
    env1_X = np.array(env1_X).reshape(int(len(env1_X)/sequence_length), sequence_length, number_of_features)
    env2_X = np.array(env2_X).reshape(int(len(env2_X)/sequence_length), sequence_length, number_of_features)
    env3_X = np.array(env3_X).reshape(int(len(env3_X)/sequence_length), sequence_length, number_of_features)

    X_train = np.concatenate((env1_X,env2_X),axis=0)
    Y_train = np.concatenate((env1_Y,env2_Y),axis=0)
    X_test = env3_X
    Y_test = env3_Y

    X_train = torch.Tensor(X_train).cuda()
    Y_train = torch.Tensor(Y_train).cuda()
    X_test = torch.Tensor(X_test).cuda()
    Y_test = torch.Tensor(Y_test).cuda()
    env1_X = torch.Tensor(env1_X)
    env1_Y = torch.Tensor(env1_Y)
    env2_X = torch.Tensor(env2_X)
    env2_Y = torch.Tensor(env2_Y)

    total_size = X_train.shape[0]
    input_size = number_of_features
    hidden_size = 100
    num_layers = 2
    num_classes = 15
    steps = 501
    lr = 0.05
    n_restarts = 1
    penalty_weight = 10
    penalty_anneal_iters = 1
    l2_weight = 0.0007
    embedding_size = math.ceil(num_classes ** 0.25)

    l1_loss = nn.L1Loss()

    for restart in range(n_restarts):
        print("Restart ", restart)
        pretty_print('step', 'Next Activity Acc', 'Timestamp Acc')
        
        model = RNN(total_size, input_size, embedding_size, hidden_size, num_layers, num_classes).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999), amsgrad=False)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 10, factor = 0.5, threshold=0.0001, min_lr = 0, cooldown=0)

        # Train the model
        for step in range(steps):
            X = env1_X
            Y = env1_Y
            X, Y = X.cuda(), Y.cuda()   
            env1_logits, env1_time = model(X)

            env1_labels = Y[:,0].type(torch.LongTensor).cuda()
            env1_loss = torch.nn.functional.cross_entropy(env1_logits, env1_labels)
            env1_penalty = penalty(env1_logits,env1_labels)

            env1_time_labels = Y[:,1].type(torch.LongTensor).cuda()
            env1_time_loss = l1_loss(env1_time, env1_time_labels)
            env1_time_penalty = time_penalty(env1_time, env1_time_labels)

            X = env2_X
            Y = env2_Y
            X, Y = X.cuda(), Y.cuda()   
            env2_logits, env2_time = model(X)

            env2_labels = Y[:,0].type(torch.LongTensor).cuda()
            env2_loss = torch.nn.functional.cross_entropy(env2_logits, env2_labels)
            env2_penalty = penalty(env2_logits,env2_labels)

            env2_time_labels = Y[:,1].type(torch.LongTensor).cuda()
            env2_time_loss = l1_loss(env2_time, env2_time_labels)
            env2_time_penalty = time_penalty(env2_time, env2_time_labels)

            train_nll = torch.stack([env1_loss, env2_loss, env1_time_loss, env2_time_loss]).mean()
            train_penalty = torch.stack([env1_penalty, env2_penalty, env1_time_penalty, env2_time_penalty]).mean()
            weight_norm = torch.tensor(0.)
            for w in model.parameters():
                weight_norm = weight_norm + w.norm().pow(2)
            
            loss = train_nll.clone()
            loss = loss + (l2_weight * weight_norm)
            penalty_weight = (penalty_weight 
                if step >= penalty_anneal_iters else 1.0)
            loss = loss + (penalty_weight * train_penalty)
            if penalty_weight > 1.0:
                # Rescale the entire loss to keep gradients in a reasonable range
                loss /= penalty_weight
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step(loss)
           
            with torch.no_grad():
                X = X_test
                Y = Y_test
                X, Y = X.cuda(), Y.cuda()
                logits, time = model(X)
                test_acc = mean_accuracy(logits, Y[:,0])
                time_test_acc = l1_loss(time, Y[:,1])
                del logits

            if step % 500 == 0:
                pretty_print(
                    np.int32(step),
                    test_acc.detach().cpu().numpy(),
                    time_test_acc.detach().cpu().numpy(),
                )

        if suffix == 'True':
            if data == 'orig':
                suffix_prediction(model, X_test, Y_test, X_train,Y_train, False)
            if data == 'gen':
                    suffix_prediction(model, X_test, Y_test, X_train,Y_train, True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='which dataset to use, gen or orig',default='gen')
    parser.add_argument('-s', '--suffix', help='compute suffix', default='False')
    args = parser.parse_args()
    run(args.data, args.suffix)

if __name__ == '__main__':
    main()
