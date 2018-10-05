import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
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
global_active_power_data = df_hourly.values
global_active_power_data = global_active_power_data.reshape((-1,1))
train_len = global_active_power_data.shape[0]*80//100

scaler = MinMaxScaler()
invscaler = MinMaxScaler()
global_active_power_data_train = global_active_power_data[:train_len+1].reshape(-1,1)
invscaler.fit(global_active_power_data_train)
global_active_power_data_train = scaler.fit_transform(global_active_power_data_train)
global_active_power_data_train = torch.from_numpy(global_active_power_data_train)

global_active_power_data_train_noscale = torch.from_numpy(global_active_power_data[1:train_len+1]).reshape(1,-1)
global_active_power_data_test = global_active_power_data[train_len+1:].reshape(-1,1)
global_active_power_data_test = torch.from_numpy(global_active_power_data_test)

predict_nmbr = global_active_power_data.shape[0]-(train_len+1)


if use_gpu:
    train_data_input = Variable(global_active_power_data_train[:train_len].reshape(1,-1)).cuda()
    train_data_output = Variable(global_active_power_data_train[1:train_len+1].reshape(1,-1)).cuda()
    test_data_output =  Variable(global_active_power_data_test.reshape(-1,1)).cuda()
#    all_data_input = Variable(global_active_power_data_all.reshape(-1,1)).cuda()
else:
    train_data_input = Variable(global_active_power_data_train[:train_len].reshape(1,-1))
    train_data_output = Variable(global_active_power_data_train[1:train_len+1].reshape(1,-1))
    test_data_output =  Variable(global_active_power_data_test.reshape(1,-1))
#    all_data_input = Variable(global_active_power_data_all.reshape(-1,1))


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


power_lstm = Sequence()
power_lstm.double()
criterion = nn.L1Loss()



backprop_steps = 25
if use_gpu:
    power_lstm.cuda()
    optimizer = optim.Adam(power_lstm.parameters(), lr = 0.01)
    for epoch in range(0,50):
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




if not use_gpu:
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
        power_lstm.load_state_dict(torch.load('training_weightshours.pt',map_location=lambda storage, loc: storage))
        fit_outputs = []
        for i in range(0,train_data_input.shape[1]):
            fit_out,h_t,c_t,h_t2,c_t2 = power_lstm(train_data_input[:,i].reshape(1,-1),h_t,c_t,h_t2,c_t2)
#            fit_out,h_t,c_t, = power_lstm(train_data_input[:,i].reshape(1,-1),h_t,c_t)
            fit_outputs += [fit_out]
        fit_outputs = torch.stack(fit_outputs, 1).squeeze(2)
        fit_outputs_numpy = fit_outputs.detach().numpy()
        fit_outputs_transformed = invscaler.inverse_transform(fit_outputs_numpy.reshape(-1,1))
        r2_lossontrain = 1 - r2_score(fit_outputs_transformed.reshape(fit_outputs_transformed.shape[0]),global_active_power_data[1:train_len+1].reshape(train_len))
        fit_outputs_torch = torch.from_numpy(fit_outputs_transformed).reshape(1,-1)
        loss = criterion(fit_outputs_torch,global_active_power_data_train_noscale)
        print("trainloss:",loss.item())
        pred = fit_outputs[:,-1].reshape(1,-1)
        all_predictions = []
        for i in range(0,predict_nmbr):
            pred,h_t,c_t,h_t2,c_t2 = power_lstm(pred,h_t,c_t,h_t2,c_t2)
#            pred,h_t,c_t = power_lstm(pred,h_t,c_t)
            all_predictions.append(pred)
        all_predictions = torch.stack(all_predictions, 1).squeeze(2).detach().numpy()
        allpredictions_inverse = all_predictions.reshape(-1,1)
        allpredictions_inverse = invscaler.inverse_transform(allpredictions_inverse) 
        all_predictions = torch.from_numpy(allpredictions_inverse.reshape(1,-1))
        loss = criterion(all_predictions,test_data_output)
        print('test loss:', loss.item())

nmbrtimes_predict=10
gap_predict=15
nmbr_predict = 3

if not use_gpu:
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
        power_lstm.load_state_dict(torch.load('training_weightshours.pt',map_location=lambda storage, loc: storage))
        allreq_forecasts = []
        for i in range(0,nmbrtimes_predict):
            data = global_active_power_data[:train_len+1+gap_predict*i]
            scaler.fit(data)
            transformed_data = scaler.transform(data)
            all_pred = []
            for j in range(0,data.shape[0]):
                pred,h_t,c_t,h_t2,c_t2 = power_lstm(torch.from_numpy(transformed_data[j]).reshape(1,-1),h_t,c_t,h_t2,c_t2)
            for k in range(0,nmbr_predict):
                pred,h_t,c_t,h_t2,c_t2 = power_lstm(pred,h_t,c_t,h_t2,c_t2)
                all_pred.append(pred)
            all_pred = torch.stack(all_pred, 1).squeeze(2)
            req_forecasts = all_pred[:,-3:].detach().numpy().reshape(-1,1)
            req_forecast = scaler.inverse_transform(req_forecasts).reshape(3).tolist()
            allreq_forecasts += req_forecast
            
            
            
trueforecasted_values = []
plt.plot(np.arange(0,150),test_data_output.detach().numpy()[0,0:150],'b')
for i in range(0,nmbrtimes_predict):
    plt.plot(np.arange(gap_predict*i,gap_predict*i+nmbr_predict),allreq_forecasts[i*nmbr_predict:(i+1)*nmbr_predict],'r')
    true_values =  test_data_output[:,gap_predict*i:gap_predict*i+3].detach().numpy()[0].tolist()
    trueforecasted_values += true_values
plt.gca().legend(('actual','forecasts'))

r2_loss_forecast = r2_score(np.array(trueforecasted_values),np.array(allreq_forecasts))

trueforecasted_valuestensor = torch.tensor(trueforecasted_values).reshape(1,-1)
allreq_forecaststensor = torch.tensor(allreq_forecasts).reshape(1,-1)
forecast_loss = criterion(trueforecasted_valuestensor,allreq_forecaststensor)

long_loss = criterion(train_data_output[:,0:10],fit_outputs_torch[:,0:10])




if not use_gpu:     

    plt.plot(all_predictions.detach().numpy()[0,0:10],'r')
    plt.plot(test_data_output.detach().numpy()[0,0:10],'b')
    plt.gca().legend(('prediction','actual'))
    

    

    
    plt.plot(global_active_power_data[0:200].reshape(200),'b')
    plt.plot(fit_outputs_torch.detach().numpy()[0,0:200],'r')
    plt.gca().legend(('actual','prediction'))


#hours
"""MAE loss forecast = 0.4836
MAE loss pred = 0.412
R2 loss pred = 0.77

MAE loss long forecast = 1.4786"""




