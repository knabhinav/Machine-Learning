
# coding: utf-8

# In[39]:


import numpy as np
import matplotlib.pyplot  as plt
import pandas as pd
from sklearn import preprocessing
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from scipy.ndimage.interpolation import shift


# In[40]:


df = pd.read_csv('household_power_consumption.txt',sep = ';')

df[["Global_active_power","Global_reactive_power","Voltage","Global_intensity","Sub_metering_1","Sub_metering_2","Sub_metering_3"]] = df[[
        "Global_active_power","Global_reactive_power","Voltage","Global_intensity","Sub_metering_1","Sub_metering_2","Sub_metering_3"]].apply(pd.to_numeric,errors='coerce')

"""As we have enough data we can remove all data for dates where there are
are cetrain values missing because removing only those rows leads in random fluctuation which is not really true"""

df.fillna(method='ffill',inplace=True)

#daily averages
del_dates = df['Date'].value_counts() [df['Date'].value_counts() < 1440].index.values
df = df[~df['Date'].isin(del_dates)]
df_daily = df.groupby("Date",sort=False)['Global_active_power'].mean()

#dataset divided hourly
df_hourly = (df["Global_active_power"].rolling(60).sum()[59::60])/60
df_hour_specific = np.reshape(df_hourly.values,(-1,24)).T


dataset_hourly = df_hourly.values
dataset_hourly = np.asarray(df_hourly)

#reshape to fit into scaler for transformation
dataset_hourly = dataset_hourly.reshape(-1,1)

#MinMax scaler
scalar = preprocessing.MinMaxScaler()
scalar.fit(dataset_hourly)
dataset_hourly = scalar.transform(dataset_hourly)



# In[43]:


#make the time series data to supervised learning
split_h = 27648

train_X_h = dataset_hourly[:split_h]
train_Y_h = dataset_hourly[1:split_h+1]

test_X_h = dataset_hourly[split_h+1:]
test_Y_h = dataset_hourly[split_h+2:]


# In[44]:


test_X_h = test_X_h[:6909]
test_Y_h = test_Y_h[:6909]


# In[45]:


#set seed
np.random.seed(200)
torch.manual_seed(200)


# In[46]:


#batch size and sequence length can be changed
#if seq =1 then its single input single output, if its another value, then its multiple input single output
BATCH_SIZE = 16
seq_len = 1
train_x = train_X_h.reshape(seq_len,-1,1)
train_y = train_Y_h.reshape(seq_len,-1,1)
test_x = test_X_h.reshape(seq_len,-1,1)
test_y = test_Y_h.reshape(seq_len,-1,1)


# In[47]:


class GRU_Model(nn.Module):
    def __init__(self):
        super(GRU_Model, self).__init__()
        self.gru1 = nn.GRUCell(1, 64)
        self.gru2 = nn.GRUCell(64, 64)
        self.linear = nn.Linear(64, 1)
        #self.drop = nn.Dropout(p=0.5)
    def forward(self, input, future = 0):
        outputs = []
        h_t = Variable(torch.zeros(input.size(1), 64).double(), requires_grad=False)#h_0 (batch, hidden_size)
        h_t2 = Variable(torch.zeros(input.size(1), 64).double(), requires_grad=False)
        for i, input_t in enumerate(input.chunk(input.size(0), dim=0)):
            #print ("Inside fwd input",input_t.size())
            input_t=input_t.squeeze(0)
            h_t = self.gru1(input_t, h_t)
            h_t2 = self.gru2(h_t, h_t2)
            output = self.linear(h_t2)
            #output = self.drop(output)
            outputs += [output]
        for i in range(future): #forecasting
            h_t = self.gru1(output, h_t)
            h_t2 = self.gru2(h_t, h_t2)
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs,1).squeeze(2)
        return outputs


# In[48]:


model = GRU_Model()
model.double()
model.load_state_dict(torch.load('MIMO_hourlyb16_ep400_hd64ad4.pth'))
# In[13]:

criterion = nn.L1Loss()
input = Variable(torch.from_numpy(train_x), requires_grad=False)
label = Variable(torch.from_numpy(train_y), requires_grad=False)
# In[]
# use LBFGS as optimizer since we can load the whole data to train
#optimizer = torch.optim.LBFGS(model.parameters(), lr=0.8)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)


# In[28]:


epochs = 300
loss_averages=[]
loss_curry = 0

for epoch in range(0,epochs):
    count=0
    for i, input_t in enumerate(input.chunk(input.size(1)//BATCH_SIZE, dim=1)):
        #print('Batch Loop', input_t.size())
        pred = model(input_t)
        target= label[:,count*BATCH_SIZE:(count+1)*BATCH_SIZE,:]
        #print('pred size',pred.size())
        #print('target size',target.size())
        target= target.squeeze(2)
        loss = criterion(pred,target.transpose(0,1))
        optimizer.zero_grad() # prepare the optimizer
        loss.backward() # compute the contribution of each parameter to the loss
        optimizer.step() # modify the parameters
        loss_curry = loss_curry + loss.item()
        count = count+1
    loss_curry = loss_curry / input.size(1)
    print("Epoch",epoch,"Loss",loss_curry)
    loss_averages.append(loss_curry)
    
torch.save(model.state_dict(),'MIMO_hourlyb16_ep100_hd64ad4Seq3.pth')


# In[49]:


results_tr = []
for i, input_t in enumerate(input.chunk(input.size(1)//BATCH_SIZE, dim=1)):
    new = model(input_t)
    results_tr.append(new)
    


# In[50]:


results_tr = torch.stack(results_tr,0)
results_tr = results_tr.transpose(1,2)
results_tr = results_tr.contiguous()
#results_tr= results_tr.squeeze(2)
results_tr= results_tr.view(-1,1)
results_tr = results_tr.data.numpy()
temp_y = train_y.reshape(-1,1)
temp_y = scalar.inverse_transform(temp_y)
results_tr = scalar.inverse_transform(results_tr)
print(r2_score(temp_y[:27647],results_tr[1:]))
print(mean_absolute_error(temp_y[:27647],results_tr[1:]))
plt.plot(results_tr[1:200],label='pred')
plt.plot(temp_y[:199],label='truth')
plt.legend(loc=2)
plt.show()


# In[51]:


results_test = []
input = Variable(torch.from_numpy(test_x[:,:6896,:]), requires_grad=False)
for i, input_t in enumerate(input.chunk(input.size(1)//1, dim=1)):
    new = model(input_t)
    results_test.append(new)
    


# In[52]:


results_test = torch.stack(results_test,0)
results_test = results_test.transpose(1,2)
results_test = results_test.contiguous()
#results_tr= results_tr.squeeze(2)
results_test= results_test.view(-1,1)
results_test = results_test.data.numpy()
temp_y = test_y.reshape(-1,1)
temp_y = scalar.inverse_transform(temp_y)
results_test = scalar.inverse_transform(results_test)
print(r2_score(temp_y[:6895],results_test[1:]))
print(mean_absolute_error(temp_y[:6895],results_test[1:]))
plt.plot(results_test[1:200],label='pred')
plt.plot(temp_y[:199],label='truth')
plt.legend(loc=2)
plt.show()


# In[ ]:


#0
future =3

data_x = dataset_hourly[:split_h+3]
data_y  = dataset_hourly[1:split_h+1+3]

train = Variable(torch.from_numpy(data_x[:-future]), requires_grad=False)
truth = Variable(torch.from_numpy(data_y), requires_grad=False)
train = train.unsqueeze(1)
result_forecast=[]
model.eval()
new = model(train,future=future)
loss= criterion(new,truth.transpose(0,1))
result_forecast= new.data.numpy()
result_forecast=result_forecast/2
rest = result_forecast.T
rest = scalar.inverse_transform(rest)
rest = rest[27648:]
data_y = scalar.inverse_transform(data_y)
plt.plot(rest,label='forecasted')
plt.plot(data_y[split_h:split_h+1+3],label='forecasted')
plt.legend(loc=2)
plt.show()

