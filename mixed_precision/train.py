from torch.autograd import Variable
import torch
import time

def train(num_epochs, cnn, loaders, loss_func, optimizer, device):

    cnn.train()
        
    # Train the model
    total_step = len(loaders['train'])
    total_time= time.time()   
    for epoch in range(num_epochs):
        
        for i, (images, labels) in enumerate(loaders['train']):
          # clear gradients for this training step   
            optimizer.zero_grad() 
            t0= time.process_time()
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images).to(device)   # batch x
            b_y = Variable(labels).to(device)   # batch y
              
            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)
            
                    
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Elapsed: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), time.process_time() - t0))
    print('total time:', time.time()-total_time)

def train_mixed_prec(num_epochs, cnn, loaders, loss_func, optimizer, device):
    scaler = torch.cuda.amp.GradScaler()

    cnn.train()
        
    # Train the model
    total_step = len(loaders['train'])
    total_time= time.time()   
    for epoch in range(num_epochs):
        
        for i, (images, labels) in enumerate(loaders['train']):
          # clear gradients for this training step   
            optimizer.zero_grad() 
            with torch.cuda.amp.autocast(): 
              t0= time.process_time()
              # gives batch data, normalize x when iterate train_loader
              b_x = Variable(images).to(device)   # batch x
              b_y = Variable(labels).to(device)   # batch y
              
              output = cnn(b_x)[0]
              loss = loss_func(output, b_y)
            
                    
            
            # backpropagation, compute gradients 
            scaler.scale(loss).backward()    
            # apply gradients             
            scaler.step(optimizer)  
            scaler.update()            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Elapsed: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), time.process_time() - t0))
    print('total time:', time.time()-total_time)


