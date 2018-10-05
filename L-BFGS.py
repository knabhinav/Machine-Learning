
# coding: utf-8

# In[ ]:


#this part should include to the main codes for LBFGS optimizer
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.8, history_size=100)

epochs = 200
loss_averages=[]
loss_curry = 0
truth= label.clone()
for epoch in range(0,epochs):
    count = 0  
    for i, input_t in enumerate(input.chunk(input.size(1)//BATCH_SIZE, dim=1)):
        target= truth[:,count*BATCH_SIZE:(count+1)*BATCH_SIZE,:]
        target= target.squeeze(2)
        #print('target size',target.size())
        def closure():
            global count
            global loss_curry
            global target
        #print('Batch Loop', input_t.size())
            pred = model(input_t)
            #print('pred size',pred.size())
            #print('target size',target.size())
            loss = criterion(pred,target.transpose(0,1))
            optimizer.zero_grad() # prepare the optimizer
            loss.backward() # compute the contribution of each parameter to the loss
            loss_curry = loss_curry + loss.item()
            
            #print("Epoch",epoch,"Loss",loss_curry)
            return loss
        count = count+1
        optimizer.step(closure) # modify the parameters
    loss_curry = loss_curry / input.size(1)
    print("Epoch",epoch,"Loss",loss_curry)
    loss_averages.append(loss_curry)
    
torch.save(model.state_dict(),'MIMO_hourlyb16_ep100_hd64LBFGs.pth')

