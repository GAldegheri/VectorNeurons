import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from models import VNAutoEncoder
from datasets import SingleModelData

def main():
    
    train_bs = 32
    #test_bs = 1000
    lr = 1e-4
    n_iter = 100
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    dataset = SingleModelData()
    n_verts = dataset.num_verts
    train_loader = DataLoader(dataset, batch_size=train_bs,
                        shuffle=False)
    train_loader = iter(train_loader)
    # test_loader = DataLoader(dataset, batch_size=test_bs,
    #                     shuffle=False)
    
    model = VNAutoEncoder(n_verts, use_relu=False)
    model.to(device)
    
    opt = Adam(model.parameters(), lr=lr)
    
    for i in range(n_iter):
        
        data = next(train_loader)
        data = data.to(device)
        
        opt.zero_grad()
        
        y_pred = model(data)
        loss = F.mse_loss(data, y_pred)
        print(i, '-', loss.item())
        loss.backward()
        opt.step()
        
if __name__=="__main__":
    main()