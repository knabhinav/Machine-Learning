
# coding: utf-8

# In[ ]:


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


# In[ ]:


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



# In[ ]:


dataset_hourly = df_hourly.values
dataset_hourly = np.asarray(df_hourly)

#reshape to fit into scaler for transformation
dataset_hourly = dataset_hourly.reshape(-1,1)

#MinMax scaler
scalar = preprocessing.MinMaxScaler()
scalar.fit(dataset_hourly)
dataset_hourly = scalar.transform(dataset_hourly)


# In[ ]:


#make the time series data to supervised learning
split_h = 27648

train_X_h = dataset_hourly[:split_h]
train_Y_h = dataset_hourly[1:split_h+1]

test_X_h = dataset_hourly[split_h+1:]
test_Y_h = dataset_hourly[split_h+2:]
#train_X_h.size


# In[ ]:


#set seed
np.random.seed(200)
torch.manual_seed(200)



# In[ ]:


BATCH_SIZE= 16 
seq_len = 1
train_x = train_X_h.reshape(seq_len,-1,1)
train_y = train_Y_h.reshape(seq_len,-1,1)
#test_x = test_X_h.reshape(seq_len,-1,1)
#test_y = test_Y_h.reshape(seq_len,-1,1)


# In[ ]:


test_X_h = test_X_h[:6896]
test_Y_h = test_Y_h[:6896]

test_x = test_X_h.reshape(seq_len,-1,1)
test_y = test_Y_h.reshape(seq_len,-1,1)
test_x.size


# In[ ]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.con1 = nn.Conv1d(1, 16, kernel_size=3, padding =2)
        self.con1b = nn.BatchNorm1d(16)
    
        self.con2 = nn.Conv1d(16,32, kernel_size=3)
        self.con2b = nn.BatchNorm1d(32)
    
        self.fc = nn.Linear(32, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 1)
        
       
        
    def forward(self, x):
        out = F.relu(self.con1b(self.con1(x)))
        #out = F.max_pool1d(out)
        out = F.relu(self.con2b(self.con2(out)))
        #out = F.max_pool1d(out)
        #print('conv2 out',out.shape)
        #print('cnn2 out',out.size())
        out = out.view(x.size(0), -1)
        
        #out = self.fc(out)
        '''out = F.dropout(F.relu(self.fc1(out)))
        out = self.fc2(out)
        '''
        return out
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        #print('num_features', num_features)
        return num_features
model = CNN()


# In[ ]:


optimizer = torch.optim.Adam(model.parameters(), lr=1e-8)

model.double()
model.load_state_dict(torch.load('CNN2tr.pth'))
# In[13]:

criterion = nn.L1Loss()
input = Variable(torch.from_numpy(train_x), requires_grad=False).double()
label = Variable(torch.from_numpy(train_y), requires_grad=False).double()


# In[ ]:


class GRU_Model(nn.Module):
    def __init__(self):
        super(GRU_Model, self).__init__()
        self.gru1 = nn.GRUCell(32, 51)
        self.gru2 = nn.GRUCell(51, 51)
        self.linear = nn.Linear(51, 1)
    def forward(self, input, future = 0):
        outputs = []
        h_t = Variable(torch.zeros(input.size(1), 51).double(), requires_grad=False)#h_0 (batch, hidden_size)
        h_t2 = Variable(torch.zeros(input.size(1), 51).double(), requires_grad=False)
        for i, input_t in enumerate(input.chunk(input.size(0), dim=0)):
            #print ("Inside fwd input",input_t.size())
            input_t=input_t.squeeze(0)
            h_t = self.gru1(input_t, h_t)
            h_t2 = self.gru2(h_t, h_t2)
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future): #forecasting
            h_t = self.gru1(output, h_t)
            h_t2 = self.gru2(h_t, h_t2)
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs,1).squeeze(2)
        return outputs

gruModel = GRU_Model()
gruModel.double()
grUoptimizer = torch.optim.Adam(gruModel.parameters(), lr=1e-5)
#gruModel.load_state_dict(torch.load('MIMO_hourlyb16_ep400_hd64.pth'))


# In[ ]:


#grUoptimizer = torch.optim.Adam(gruModel.parameters(), lr=1e-4)
epochs = 200 
loss_averages=[]
loss_curry = 0
model.train()
for epoch in range(0,epochs):
    count=0
    for i, input_t in enumerate(input.chunk(input.size(1)//BATCH_SIZE, dim=1)):
        input_t = input_t.transpose(0,1)
        input_t = input_t.transpose(1,2)
        #print('Batch Loop', input_t.size())
        #print(input_t[0,0,-1])
        pred_cnn = model(input_t)
        pred_cnn = pred_cnn.unsqueeze(0)
        #print('pred cnn size', pred_cnn.size())
        pred_gru = gruModel(pred_cnn)
        #pred_gru = pred_gru.narrow(0,0,16)
        pred_gru = pred_gru.unsqueeze(1)
        #pred_gru = pred_gru.squeeze(2)
        target= label[:,count*BATCH_SIZE:(count+1)*BATCH_SIZE,:]
        #print('pred gru size',pred_gru.size(),pred_gru)
        #print('target size',target.size())
        #target= target.squeeze(2)
        #target = target.transpose(0,1)
        #print(target[0,0])
        #print(target.size())
        loss = criterion(pred_gru,target.transpose(0,1))
        grUoptimizer.zero_grad() # prepare the gru optimizer
        optimizer.zero_grad() # prepare the cnn optimizer
        loss.backward() # compute the contribution of each parameter to the loss
        grUoptimizer.step() # modify the parameters
        optimizer.step()
        loss_curry = loss_curry + loss.item()
        count = count+1
    loss_curry = loss_curry / input.size(1)
    print("Epoch",epoch,"Loss",loss_curry)
    loss_averages.append(loss_curry)
#torch.save(gruModel.state_dict(),'gru_CNN2tr2.pth')


# In[ ]:


gruModel.load_state_dict(torch.load('gru_CNN2tr2.pth'))
model.eval()
results_t = []
for i, input_t in enumerate(input.chunk(input.size(1)//BATCH_SIZE, dim=1)):
    input_t = input_t.transpose(0,1)
    input_t = input_t.transpose(1,2)
    pred_cnn = model(input_t)
    pred_cnn = pred_cnn.unsqueeze(0)
    pred_gru = gruModel(pred_cnn)
    results_t.append(pred_gru)


# In[ ]:


results_tr = torch.stack(results_t,0)
#results_tr = results_tr.transpose(1,2)
#results_tr = results_tr.contiguous()
#results_tr= results_tr.squeeze(2)
results_tr= results_tr.view(-1,1)
results_tr = results_tr.data.numpy()
temp_y = train_y.reshape(-1,1)
#temp_ = temp_y[0::3]
#temp_y = scalar.inverse_transform(temp_y)
#results_tr = scalar.inverse_transform(results_tr)
print(r2_score(temp_y[:27647],results_tr[1:]))
print(mean_absolute_error(temp_y[:27647],results_tr[1:]))


# In[ ]:


plt.plot(results_tr[1:200],label='pred')
plt.plot(temp_y[:199],label='truth')
plt.legend(loc=2)
plt.savefig('CNNGRUtr200inver.png')
plt.show()


# In[ ]:


model.eval()
results_test = []
input = Variable(torch.from_numpy(test_x[:,:,:]), requires_grad=False)
for i, input_t in enumerate(input.chunk(input.size(1)//BATCH_SIZE, dim=1)):
    input_t = input_t.transpose(0,1)
    input_t = input_t.transpose(1,2)
    pred_cnn = model(input_t)
    pred_cnn = pred_cnn.unsqueeze(0)
    pred_gru = gruModel(pred_cnn)
    results_test.append(pred_gru)
    


# In[ ]:


len(results_test)


# In[ ]:


results_test = torch.stack(results_test,0)
#results_test = results_test.transpose(1,2)
#results_test = results_test.contiguous()
#results_tr= results_tr.squeeze(2)
results_test= results_test.view(-1,1)
results_test = results_test.data.numpy()
temp_yt = test_y.reshape(-1,1)
#temp_yt = scalar.inverse_transform(temp_yt)
#results_test = scalar.inverse_transform(results_test)
print(r2_score(temp_yt[:6895],results_test[1:]))
plt.plot(results_test,label='pred')
plt.plot(temp_yt,label='truth')
plt.legend(loc=2)
#plt.savefig("testB16GRUCNN.png")
plt.show()


# In[ ]:


plt.plot(results_test[1:200],label='pred')
plt.plot(temp_yt[:199],label='truth')
plt.legend(loc=2)
#plt.savefig('CNNGRUtest200.png')
plt.show()


# In[ ]:


print(mean_absolute_error(temp_y[:27647],results_tr[1:]))
print(r2_score(temp_yt[:6895],results_test[1:]))
print(mean_absolute_error(temp_yt[:6895],results_test[1:]))

