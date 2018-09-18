import sys
from collections import defaultdict
import itertools

import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from sklearn import metrics
import numpy as np
import json
import torchtext.vocab

import data.gen_one_hot

NUM_EPOCHS = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = "data//train.json"
dev_data = "data//dev.json"
test_data = "data//test.json"
all_data = "data//final_data.json"

# We will generate our own one-hot embeddings and throw out the ones that are already in the .json files, if there are any
g = data.gen_one_hot.Gen_data(all_data)

# all the one-hot embedding etc. has to be handled here, rather in data generation,
# because we have to know which word indice go to which real words in order to do the embedding.

#--------------- Model --------------
class Neural_Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Neural_Net, self).__init__()
        self.l1 = nn.Linear(input_size, 70, bias = True)
        self.l1_act = nn.Tanh()
        self.l2 = nn.Linear(70,35, bias = True)
        self.l2_act = nn.Tanh()
        self.l3 = nn.Linear(35, output_size, bias = True)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        out = x
        out = self.l1(out)
        out = self.l1_act(out)
        out = self.l2(out)
        out = self.l2_act(out)
        out = self.l3(out)
        out = self.log_softmax(out)
        return out


#------------------ Loss & Optimizer --------------------
def cross_entropy_loss(logp_hats, ps):
    return torch.mean(torch.sum(- ps * logp_hats, 1))

#------------ Training The Model -----------------------
def train(model, optimizer, loss_fn, training_data, num_epochs):
    for epoch in range(num_epochs):
        for i, env_data in enumerate(training_data):
            x = Variable(torch.Tensor(env_data["x"]))
            y = Variable(torch.Tensor(env_data["y"]))
            
            log_y_pred = model(x)
            
            optimizer.zero_grad()
            loss = loss_fn(log_y_pred, y)
            print(epoch, loss.item(), end='\r')

            loss.backward()
            
            optimizer.step()

#-------------- Metrics ---------------------
def cross_entropy(log_pred, targets):
    log_pred = Variable(torch.Tensor(log_pred))
    targets = Variable(torch.Tensor(targets))

    result = sum(torch.sum(-targets * log_pred, 1).data.numpy()) #I should as Professor Singh for help here as torch.sum is returning an array over the 6 instances. Does summing those before returning them make sense? And does it make sense to take the average once all those instances have been collected (as shown in line 86). Right now it doesn't make sense because cross entropy is really high compared to mse
    return result

def accuracy(pred, targets):
    return np.argmax(pred) == np.argmax(targets)

# Embedding logic

def read_data_line(line, embedding=None):
    env_data = eval(line)
    # Construct x out of its parts; throw away any existing x field
    target = env_data["target"]
    env_objs = env_data["env"]
    utter = env_data["utter"]
    zipped_objs = list(zip(env_objs, target)) # Are there biases in ordering that the network might be picking up on?
    _, env_vec, _ = g.gen_env_vec(zipped_objs)
    if embedding:
        utterance_vec = embedding.vectors[embedding.stoi[utter]]
    else:
        utterance_vec = g.build_one_hot_utter(utter)
    env_data['x'] = [np.concatenate((utterance_vec, env_vec), axis=0).tolist()]
    return env_data

def classify(env_data):
    utterance = env_data['utter']
    obj1 = env_data['env'][0]
    obj2 = env_data['env'][1]
    obj3 = env_data['env'][2]
    properties = ['shape', 'color']
    if utterance == obj1['shape'] == obj2['shape'] == obj3['shape']:
        return "3-ambiguous"
    elif utterance == obj1['color'] == obj2['color'] == obj3['color']:
        return "3-ambiguous"
    for prop, otherprop in itertools.permutations(properties):
        for obj1, obj2, obj3 in itertools.permutations([obj1, obj2, obj3]):
            if utterance == obj1[prop] == obj2[prop]:
                if obj1[otherprop] == obj3[otherprop] or obj2[otherprop] == obj3[otherprop]:
                    return "2-pragmatic"
                else:
                    return "2-ambiguous"
    return "unclassified"

def print_evaluation(model, embedding, data_filename, preds_filename, name, show_failure_cases=False):
    count = 0
    aggregated_acc = 0
    aggregated_MSE = 0
    aggregated_CE = 0

    type_totals = defaultdict(list)
    type_wrongs = defaultdict(list)

    with open(preds_filename, 'w') as out:
        for env_data in open(data_filename, 'r'):
            env_data = read_data_line(env_data, embedding)
            x = Variable(torch.Tensor(env_data["x"]))
      
            log_y_hat = model(x).data.numpy()
            y = env_data["y"]

            aggregated_CE += cross_entropy(log_y_hat,y)

            example_type = classify(env_data)
            type_totals[example_type].append(env_data)
            
            for log_p_hat,p in zip(log_y_hat, y):
                accurate = accuracy(log_p_hat, p)
                if not accurate:
                    type_wrongs[example_type].append(env_data)
                    if show_failure_cases:
                        print(env_data['utter'], env_data['env'], env_data['y'], np.exp(log_y_hat))
                aggregated_acc += accurate
                aggregated_MSE += metrics.mean_squared_error(np.exp(log_p_hat), p)
                count += 1
            env_data["y_hat"] = np.exp(log_y_hat).tolist()
            json.dump(env_data, out)
            out.write('\n')


    avg_acc = aggregated_acc/count
    avg_MSE = aggregated_MSE/count
    avg_CE = aggregated_CE/count
        
    print(name, "avg acc: ", avg_acc)
    print(name, "avg mse", avg_MSE)
    print(name, "avg ce", avg_CE)

    for example_type in type_totals:
        print(name, "failures on example type %s:" % example_type, len(type_wrongs[example_type]), "/", len(type_totals[example_type]))
        

def main(embedding=None):
    if embedding == 'glove':
        embedding = torchtext.vocab.GloVe('6B', dim=100)
    elif embedding is not None:
        print("Unknown embedding: %s" % embedding, file=sys.stderr)
        print("Aborting.", file=sys.stderr)
        sys.exit(1)

    with open(all_data) as infile:
        ex_inst = read_data_line(infile.readline(), embedding=embedding)
    input_size, output_size = len(ex_inst["x"][0]),len(ex_inst["y"][0])
    
    model = Neural_Net(input_size, output_size).to(device)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print("name: %s, param_data: %s"% (name, param.data))

    loss_fn = cross_entropy_loss #this can be switched with torch.nn.MSELoss

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = .0001)

    with open(train_data, 'r') as train_data_file:
        training_data = [read_data_line(line, embedding) for line in train_data_file]
    train(model, optimizer, loss_fn, training_data, num_epochs=NUM_EPOCHS)
    
        
    print_evaluation(model, embedding, train_data, "train_preds.json", "training", show_failure_cases=False)
    print_evaluation(model, embedding, dev_data, "dev_preds.json", "dev", show_failure_cases=False)    


if __name__ == '__main__':
    main(*sys.argv[1:])








