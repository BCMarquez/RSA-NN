import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from sklearn import metrics
import numpy as np
import json


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_data = "data//train.json"
dev_data = "data//dev.json"
test_data = "data//test.json"
data = "data//final_data.json"

ex_inst = eval(open(data).readline())
input_size, output_size = len(ex_inst["x"][0]),len(ex_inst["y"][0])

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
        out = self.l1(x)
        out = self.l1_act(out)
        out = self.l2(out)
        out = self.l2_act(out)
        out = self.l3(out)
        out = self.log_softmax(out)
        return out

model = Neural_Net(input_size, output_size).to(device)

for name, param in model.named_parameters():
    if param.requires_grad:
        print("name: %s, param_data: %s"% (name, param.data))



#------------------ Loss & Optimizer --------------------
def cross_entropy_loss(logp_hats, ps):
    return torch.mean(torch.sum(- ps * logp_hats, 1))

loss_fn = cross_entropy_loss #this can be switched with torch.nn.MSELoss

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = .0001)

#------------ Training The Model -----------------------
for epoch in range(30):
    for i,env_data in enumerate(open(train_data,'r')):
        env_data = eval(env_data)
        x = Variable(torch.Tensor(env_data["x"]))
        y = Variable(torch.Tensor(env_data["y"]))

        log_y_pred = model(x)

        optimizer.zero_grad()
        loss = loss_fn(log_y_pred, y)
        print(epoch, loss.item(), end='\r')


        loss.backward()

        optimizer.step()
    time.sleep(1)

#-------------- Metrics ---------------------
def cross_entropy(log_pred, targets):
    log_pred = Variable(torch.Tensor(log_pred))
    targets = Variable(torch.Tensor(targets))

    result = sum(torch.sum(-targets * log_pred, 1).data.numpy()) #I should as Professor Singh for help here as torch.sum is returning an array over the 6 instances. Does summing those before returning them make sense? And does it make sense to take the average once all those instances have been collected (as shown in line 86). Right now it doesn't make sense because cross entropy is really high compared to mse
    return result

def accuracy(pred, targets):
    return np.argmax(pred) == np.argmax(targets)
        

#------------ Training Data Metrics ----------------------
count = 0
aggregated_acc = 0
aggregated_MSE = 0
aggregated_CE = 0

with open('train_preds.json','a') as out:
    for env_data in open(train_data, 'r'):
        env_data = eval(env_data)
        x = Variable(torch.Tensor(env_data["x"]))
      
        log_y_hat = model(x).data.numpy()
        y = env_data["y"]

        aggregated_CE += cross_entropy(log_y_hat,y)

        for log_p_hat,p in zip(log_y_hat, y):
            aggregated_acc += accuracy(log_p_hat, p)
            aggregated_MSE += metrics.mean_squared_error(np.exp(log_p_hat), p)
            count += 1
        env_data["y_hat"] = np.exp(log_y_hat).tolist()
        json.dump(env_data,out)
        out.write('\n')


avg_acc = aggregated_acc/count
avg_MSE = aggregated_MSE/count
avg_CE = aggregated_CE/count
        
print("training avg acc: ", avg_acc)
print("training avg mse", avg_MSE)
print("training avg ce", avg_CE)
        
#------------- Dev Data Metrics -----------------------
count = 0
aggregated_acc = 0
aggregated_MSE = 0
aggregated_CE = 0

with open('dev_preds.json', 'a') as out:
    for env_data in open(dev_data, 'r'):
        env_data = eval(env_data)
        x = Variable(torch.Tensor(env_data["x"]))
      

        log_y_hat = model(x).data.numpy()
        y = env_data["y"]

        aggregated_CE += cross_entropy(log_y_hat,y)

        for log_p_hat,p in zip(log_y_hat, y):
            aggregated_acc += accuracy(log_p_hat, p)
            aggregated_MSE += metrics.mean_squared_error(np.exp(log_p_hat), p)
            count += 1
        env_data["y_hat"] = np.exp(log_y_hat).tolist()
        json.dump(env_data,out)
        out.write('\n')

avg_acc = aggregated_acc/count
avg_MSE = aggregated_MSE/count
avg_CE = aggregated_CE/count
        
print("dev avg acc: ", avg_acc)
print("dev avg mse", avg_MSE)
print("dev avg ce", avg_CE)

#------------- Test Data Metrics ------------------
#will be added once I tune my hyper parameters well.





