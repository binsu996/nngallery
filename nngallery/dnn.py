import torch
import numpy as np

class BasicRegressor(torch.nn.Module):
    def __init__(self,hiddenlayers,hidden_size):
        super().__init__()
        self.hiddenlayers=hiddenlayers
        self.output=torch.nn.Linear(hidden_size,1)
    
    def forward(self,x,y=None):
        x=self.hiddenlayers(x)
        x=self.output(x)
        if y is not None:
            loss=((y.reshape(-1,1)-x)**2).mean()
            return loss
        else:
            return x.reshape(-1)

    def train(self,train_X,train_y,epoch=16,cuda=True,batch_size=128,lr=1e-4):
        opt=torch.optim.Adam(self.parameters(),lr=lr)
        train_dataset=torch.utils.data.TensorDataset(torch.Tensor(train_X),torch.Tensor(train_y))
        train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

        if cuda:
            self.cuda()
        
        for i in range(1,epoch+1):
            for x,y in train_dataloader:
                if cuda:
                    x,y=x.cuda(),y.cuda()
                opt.zero_grad()
                loss=self.forward(x,y)
                loss.backward()
                opt.step()
                print(f'{i}-th loss: {loss.item():.3f}')
        
        self.cpu()
        
    def predict(self,test_X,batch_size=None,cuda=True):

        if batch_size is not None:
            raise NotImplementedError
        if cuda:
            self.cuda()
        with torch.no_grad():
            x=torch.Tensor(test_X)
            if cuda:
                x=x.cuda()
            prediction=self.forward(x)

        self.cpu()

        return prediction.cpu().numpy()

class BasicClassifier(torch.nn.Module):
    def __init__(self,output_size,hiddenlayers,hidden_size):
        super().__init__()
        self.hiddenlayers=hiddenlayers
        self.output_size=output_size
        self.output=torch.nn.Linear(hidden_size,output_size)
    
    def forward(self,x,y=None):
        x=self.hiddenlayers(x)
        x=self.output(x)
        if y is not None:
            loss=torch.nn.CrossEntropyLoss()(x,y.long())
            return loss
        else:
            return x.argmax(dim=1)

    def train(self,train_X,train_y,epoch=16,cuda=True,batch_size=128,lr=1e-4):
        opt=torch.optim.Adam(self.parameters(),lr=lr)
        train_dataset=torch.utils.data.TensorDataset(torch.Tensor(train_X),torch.Tensor(train_y))
        train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

        if cuda:
            self.cuda()
        
        for i in range(1,epoch+1):
            for x,y in train_dataloader:
                if cuda:
                    x,y=x.cuda(),y.cuda()
                opt.zero_grad()
                loss=self.forward(x,y)
                loss.backward()
                opt.step()
                print(f'{i}-th loss: {loss.item():.3f}')
        
        self.cpu()
        

    def predict(self,test_X,batch_size=None,cuda=True):

        if batch_size is not None:
            raise NotImplementedError
        if cuda:
            self.cuda()
        with torch.no_grad():
            x=torch.Tensor(test_X)
            if cuda:
                x=x.cuda()
            prediction=self.forward(x)

        self.cpu()

        return prediction.cpu().numpy()
        

        
                  


        

            


