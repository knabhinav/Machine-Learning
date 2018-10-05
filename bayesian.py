# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 13:46:03 2018

@author: abhi
"""
import GPy
import optunity
import sobol
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm

use_gpu = torch.cuda.is_available()
df = pd.read_csv('household_power_consumption.txt',sep = ';')
df[["Global_active_power","Global_reactive_power","Voltage","Global_intensity","Sub_metering_1","Sub_metering_2","Sub_metering_3"]] =df[[
        "Global_active_power","Global_reactive_power","Voltage","Global_intensity","Sub_metering_1","Sub_metering_2","Sub_metering_3"]].apply(pd.to_numeric,errors='coerce')
"""As we have enough data we can remove all data for dates where there are
are cetrain values missing because removing only those rows leads in random fluctuation which is not really true"""
df.fillna(method='ffill',inplace=True)

#daily averages
del_dates = df['Date'].value_counts() [df['Date'].value_counts() < 1440].index.values
df = df[~df['Date'].isin(del_dates)]
df_daily = df.groupby("Date",sort=False)['Global_active_power'].mean()
#lstm is expected to capture mainly seosanal differences  and more power on weekend and festive days also 

#dataset divided hourly
df_hourly = (df["Global_active_power"].rolling(60).sum()[59::60])/60
df_hour_specific = np.reshape(df_hourly.values,(-1,24)).T
#lstm is expected to capture seosonal differences rises in power on weekend days showing jerks on weekends and 

#normal dataset
df_small = df.set_index('Date')
df_small = df_small.loc['1/1/2007':'1/1/2009',:]
#lstm is expcted to capture rises in peak hours and also increses in weekends. Will not capture seosonal diff becuases of vanishing gradients.
    
np.random.seed(0)
torch.manual_seed(0)
global_active_power_data = df_daily.values
global_active_power_data = global_active_power_data.reshape((-1,1))
train_len = global_active_power_data.shape[0]*80//100

scaler = MinMaxScaler()
invscaler = MinMaxScaler()
global_active_power_data_train = global_active_power_data[:train_len+1].reshape(-1,1)
invscaler.fit(global_active_power_data_train)
global_active_power_data_train = scaler.fit_transform(global_active_power_data_train)
global_active_power_data_train = torch.from_numpy(global_active_power_data_train)


global_active_power_data_test = global_active_power_data[train_len+1:].reshape(-1,1)
global_active_power_data_test = torch.from_numpy(global_active_power_data_test)

predict_nmbr = global_active_power_data.shape[0]-(train_len+1)


if use_gpu:
    train_data_input = Variable(global_active_power_data_train[:train_len].reshape(1,-1)).cuda()
    train_data_output = Variable(global_active_power_data_train[1:train_len+1].reshape(1,-1)).cuda()
    test_data_output =  Variable(global_active_power_data_test.reshape(-1,1)).cuda()
else:
    train_data_input = Variable(global_active_power_data_train[:train_len].reshape(1,-1))
    train_data_output = Variable(global_active_power_data_train[1:train_len+1].reshape(1,-1))
    test_data_output =  Variable(global_active_power_data_test.reshape(1,-1))

#data = test_data_output.detach().numpy().reshape(1,-1)


hidden_featrues = 125
class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, hidden_featrues)
        self.lstm2 = nn.LSTMCell(hidden_featrues,hidden_featrues)
        self.linear = nn.Linear(hidden_featrues, 1)
    def forward(self, input, h_t,c_t,h_t2,c_t2):
        h_t, c_t = self.lstm1(input, (h_t, c_t))
        h_t2,c_t2 = self.lstm2(h_t,(h_t2,c_t2))
        output = self.linear(h_t2)
        return [output,h_t, c_t,h_t2,c_t2]



criterion = nn.L1Loss()

backprop_stepss = []
import sobol
number_of_samples = 30
parameterUpperLimits = np. array ([20])
parameterLowerLimits = np. array ([100])
for i in range ( number_of_samples ):
    x = sobol . i4_sobol (1,i)[0] * ( parameterUpperLimits -parameterLowerLimits ) + parameterLowerLimits           
    backprop_stepss.append(x) 

backprop_stepss = np.concatenate(backprop_stepss,axis=0)


def lstm_network(backprop_steps,train_data_input):
    if use_gpu:
        power_lstm = Sequence()
        power_lstm.double()
        power_lstm.cuda()
        optimizer = optim.Adam(power_lstm.parameters(), lr = 0.01)
        for epoch in range(0,25):
            if use_gpu:
                h_t = Variable(torch.zeros(train_data_input.shape[0], hidden_featrues, dtype=torch.double)).cuda()
                c_t = Variable(torch.zeros(train_data_input.shape[0], hidden_featrues, dtype=torch.double)).cuda()
                h_t2 = Variable(torch.zeros(train_data_input.shape[0], hidden_featrues, dtype=torch.double)).cuda()
                c_t2 = Variable(torch.zeros(train_data_input.shape[0], hidden_featrues, dtype=torch.double)).cuda()
            else:
                h_t = Variable(torch.zeros(train_data_input.shape[0], hidden_featrues, dtype=torch.double))
                c_t = Variable(torch.zeros(train_data_input.shape[0], hidden_featrues, dtype=torch.double))
                h_t2 = Variable(torch.zeros(train_data_input.shape[0], hidden_featrues, dtype=torch.double))
                c_t2 = Variable(torch.zeros(train_data_input.shape[0], hidden_featrues, dtype=torch.double))
            outputs = []
            for i in range(0,train_data_input.shape[1]):
                out,h_t,c_t,h_t2,c_t2 = power_lstm(train_data_input[:,i].reshape(1,-1),h_t,c_t,h_t2,c_t2)
#            out,h_t,c_t = power_lstm(train_data_input[:,i].reshape(1,-1),h_t,c_t)
                outputs += [out]
                if (i+1)%backprop_steps == 0:
                    optimizer.zero_grad()
                    outputs = torch.stack(outputs, 1).squeeze(2)
                    loss = criterion(outputs,train_data_output[:,(int((i+1)//backprop_steps)-1)*backprop_steps:i+1])
                    print("loss:",loss)
                    loss.backward()
                    out.detach_()
                    h_t.detach_()
                    c_t.detach_()
                    h_t2.detach_()
                    c_t2.detach_()
                    outputs = []
                    optimizer.step()
        torch.save(power_lstm.state_dict(),'training_weights.pt')
    if use_gpu:
        with torch.no_grad():
            if use_gpu:
                h_t = Variable(torch.zeros(train_data_input.shape[0], hidden_featrues, dtype=torch.double)).cuda()
                c_t = Variable(torch.zeros(train_data_input.shape[0], hidden_featrues, dtype=torch.double)).cuda()
                h_t2 = Variable(torch.zeros(train_data_input.shape[0], hidden_featrues, dtype=torch.double)).cuda()
                c_t2 = Variable(torch.zeros(train_data_input.shape[0], hidden_featrues, dtype=torch.double)).cuda()
            else:
                h_t = Variable(torch.zeros(train_data_input.shape[0], hidden_featrues, dtype=torch.double))
                c_t = Variable(torch.zeros(train_data_input.shape[0], hidden_featrues, dtype=torch.double))
                h_t2 = Variable(torch.zeros(train_data_input.shape[0], hidden_featrues, dtype=torch.double))
                c_t2 = Variable(torch.zeros(train_data_input.shape[0], hidden_featrues, dtype=torch.double))
            power_lstm.load_state_dict(torch.load('training_weights.pt',map_location=lambda storage, loc: storage))
            fit_outputs = []
            for i in range(0,train_data_input.shape[1]):
                fit_out,h_t,c_t,h_t2,c_t2 = power_lstm(train_data_input[:,i].reshape(1,-1),h_t,c_t,h_t2,c_t2)
#            fit_out,h_t,c_t, = power_lstm(train_data_input[:,i].reshape(1,-1),h_t,c_t)
                fit_outputs += [fit_out]
            fit_outputs = torch.stack(fit_outputs, 1).squeeze(2)
            loss = criterion(fit_outputs, train_data_output)
            return loss.item()

all_losses = []
for backprop_steps in backprop_stepss:
    loss = lstm_network(backprop_steps,train_data_input)
    all_losses.append(loss.item())

"""all_losses = np.array(all_losses)


kern = GPy.kern.RBF(1) + GPy.kern.White(1)
backprop_stepss = backprop_stepss.reshape(-1,1)

model = GPy.models.GPRegression(backprop_stepss, all_losses, kernel=kern)

class expected_improvement:
    def __init__(self, E_best, model):
        self.E_best = E_best
        self.model = model
        
    def foo(self, x):
        E_best = self.E_best
        model = self.model
        var_q = model.predict(np.array([[x]]))[1][0][0]
        E_mean = model.predict(np.array([[x]]))[0][0][0]
        q = np.asarray([x])
        return np.sqrt(var_q) * (gamma(E_mean, E_best, var_q) * Phi(gamma(E_mean, E_best, var_q)) + phi(gamma(E_mean, E_best, var_q)))

def gamma(E_mean, E_best, var_q):
    # E_best is the current best guess of the target function's minimal value
    return (E_best - E_mean)/np.sqrt(var_q)

def Phi(q):
    return norm.cdf(q)

def phi(q):
    return norm.pdf(q)

def maximize_expected_improvement(E, model, print_=False):
    e = expected_improvement(np.min(E), model)
    #E_best = np.min(E)
    minimum = optunity.minimize(e.foo, x = [20,100])
    x = minimum[0]['x']
    #print(maximum)
    if print_ == True:
        print('x: ', x)
    return x

x_max = maximize_expected_improvement(all_losses, model, print_=True)



def evaluate_new_point_and_add_to_training_set(q, Q_input, E):
    x = q[0]
    # add new point to trainingsset
    Q_input = Q_input.tolist()
    Q_input.append([x])
    E = E.tolist()
    E.append([lstm_network(x,train_data_input)])
    Q_input = np.asarray(Q_input)
    E = np.asarray(E)
    return Q_input, E


def repeat_steps(Q_input, E_,N = 10):
    kern = GPy.kern.RBF(1) + GPy.kern.White(1)
    print(np.shape(Q_input), np.shape(E_))
    for i in range(N):
        model = GPy.models.GPRegression(Q_input, E_, kernel=kern)
        x_max= maximize_expected_improvement(E_, model)
        Q_input, E_ = evaluate_new_point_and_add_to_training_set([x_max], Q_input, E_)
        a = E_
    return [Q_input,E_]

backprop_stepss,all_losses = repeat_steps(backprop_stepss, all_losses)"""

plt.plot(df_hourly.values)
plt.title('Average active global power(hourly)')

