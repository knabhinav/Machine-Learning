
# coding: utf-8

# In[1]:


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


# In[2]:


#load data
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


# In[3]:


dataset_daily = df_daily.values
dataset_daily = np.asarray(dataset_daily)

#reshape to fit into scaler for transformation
dataset_daily = dataset_daily.reshape(-1, 1)

#MinMax scaler
scalar = preprocessing.MinMaxScaler()
scalar.fit(dataset_daily)
dataset_daily = scalar.transform(dataset_daily)


# In[4]:


#make the time series data to supervised learning
split_d = 1152

train_X_d = dataset_daily[:split_d]
train_Y_d = dataset_daily[1:split_d+1]

test_X_d = dataset_daily[split_d+1:]
test_Y_d = dataset_daily[split_d+2:]
test_Y_d = test_Y_d[:-1]


# In[5]:


################################################  daily ##########################################

#set seed
np.random.seed(200)
torch.manual_seed(200)


# In[6]:


#batch size and seq length can be changed, batch size16 performs best
#if seq =1 then single input single output is given, if it is different
#multiple input single output is given
BATCH_SIZE= 1
seq_len = 1
train_x = train_X_d.reshape(seq_len,-1,1)
train_y = train_Y_d.reshape(seq_len,-1,1)
test_x = test_X_d.reshape(seq_len,-1,1)
test_y = test_Y_d.reshape(seq_len,-1,1)


# In[7]:


#hidden size can be changed, hd =64 is the best
class GRU_Model(nn.Module):
    def __init__(self,batch):
        super(GRU_Model, self).__init__()
        self.gru1 = nn.GRUCell(1, 64)
        self.gru2 = nn.GRUCell(64, 64)
        self.linear = nn.Linear(64, 1)
        self.batch_size=batch
    def forward(self, input, future =0): #if future is different than 0, it is another method for future forecasting
        outputs = []
        h_t = Variable(torch.zeros(self.batch_size, 64).double(), requires_grad=False)#h_0 (batch, hidden_size)
        h_t2 = Variable(torch.zeros(self.batch_size, 64).double(), requires_grad=False)
        for i, input_t in enumerate(input.chunk(input.size(0), dim=0)):
            input_t=input_t.squeeze(0)
            #print ("Inside fwd input",input_t.size())
            h_t = self.gru1(input_t, h_t)
            h_t2 = self.gru2(h_t, h_t2)
            output = self.linear(h_t2)
            outputs += [output]

        #uncomment following for future forecasting t+3 if future is equal to 3 #better method
        #for i in range(future): #forecasting
        #    h_t = self.gru1(output, h_t)
        #    h_t2 = self.gru2(h_t, h_t2)
        #    output = self.linear(h_t2)
        #    outputs += [output]
        outputs = torch.stack(outputs,1).squeeze(2)
        #print ("Inside fwd output",outputs.size())
        return outputs


# In[12]:


model = GRU_Model(BATCH_SIZE)
model.double()
criterion = nn.MSELoss()
criterion = nn.L1Loss()
# use LBFGS as optimizer since we can load the whole data to train
#optimizer = torch.optim.LBFGS(model.parameters(), lr=0.8)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#model.load_state_dict(torch.load('MIMO_daily_b1seq1_ep400_hd64.pth'))

input = Variable(torch.from_numpy(train_x), requires_grad=False)
label = Variable(torch.from_numpy(train_y), requires_grad=False)
#change epoch number (400is good)
epochs = 100
loss_averages=[]
loss_curry = 0

#train
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
torch.save(model.state_dict(),'MIMO_daily_b1seq1_ep400_hd64.pth')


# In[14]:


#train set
results_tr = []
for i, input_t in enumerate(input.chunk(input.size(1)//BATCH_SIZE, dim=1)):
    new = model(input_t)
    results_tr.append(new)
    


# In[15]:


results_tr = torch.stack(results_tr,0)
results_tr = results_tr.transpose(1,2)
results_tr = results_tr.contiguous()
#results_tr= results_tr.squeeze(2)
results_tr= results_tr.view(-1,1)
results_tr = results_tr.data.numpy()
temp_y = train_y.reshape(-1,1)

#inverse transform
temp_y = scalar.inverse_transform(temp_y)
results_tr = scalar.inverse_transform(results_tr)
#r2score and plot
print(r2_score(temp_y[:1151],results_tr[1:]))
plt.plot(results_tr,label='pred')
plt.plot(temp_y,label='truth')
plt.savefig("trainB1daily.png")
plt.legend(loc=2)
plt.show()


# In[16]:


#for 200 points
plt.plot(results_tr[1:200],label='pred')
plt.plot(temp_y[:199],label='truth')
plt.savefig("traindaily200.png")
plt.legend(loc=2)
plt.show()


# In[17]:


print("R2 daily train, batch1 ", r2_score(temp_y[:1151],results_tr[1:]))
print("MAE daily train, batch1 ", mean_absolute_error(temp_y[:1151],results_tr[1:]))


# In[18]:


results_te = []
#if batch size is given, then for testing the number should be given, which can be divided by batch size
#for ex for batch=16 num should be equal to 272
num = test_x.size
input = Variable(torch.from_numpy(test_x[:,:num,:]), requires_grad=False)
for i, input_t in enumerate(input.chunk(input.size(1)//BATCH_SIZE, dim=1)):
    new = model(input_t)
    results_te.append(new)
    


# In[19]:


results_te = torch.stack(results_te,0)
results_te = results_te.transpose(1,2)
results_te = results_te.contiguous()
#results_tr= results_tr.squeeze(2)
results_te = results_te.view(-1,1)
results_te = results_te.data.numpy()
temp_y = test_y.reshape(-1,1)
#inverse
temp_y = scalar.inverse_transform(temp_y)
results_te = scalar.inverse_transform(results_te)
print(r2_score(temp_y[:286],results_te[1:286]))
print(mean_absolute_error(temp_y[:286], results_te[1:286]))
plt.plot(results_te,label='pred')
plt.plot(temp_y,label='truth')
plt.savefig("TESTB1daily.png")
plt.legend(loc=2)
plt.show()

