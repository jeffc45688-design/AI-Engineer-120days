import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import transforms,datasets
from torch.utils.data import DataLoader,random_split
from models.resnet_model import get_model
import config
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
dataset=datasets.ImageFolder(config.DATA_PATH,transform=transform)
train_size=int(0.8*len(dataset))
val_size=len(dataset)-train_size
train_set,val_set=random_split(dataset,[train_size,val_size])
train_loader=DataLoader(train_set,batch_size=config.BATCH_SIZE,shuffle=True)
val_loader=DataLoader(val_set,batch_size=config.BATCH_SIZE)
model=get_model()
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.fc.parameters(),lr=config.LR)
train_losses=[]
val_losses=[]
val_accs=[]
best_val_loss=float("inf")
for epoch in range(config.EPOCHS):
    model.train()
    total_train_loss=0
    for X,Y in train_loader:
        output=model(X)
        loss=criterion(output,Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss+=loss.item()
    avg_train_loss=total_train_loss/len(train_loader)
    model.eval()
    total_val_loss=0
    correct=0
    total=0
    with torch.no_grad():
        for X,Y in val_loader:
            output=model(X)
            loss=criterion(output,Y)
            total_val_loss+=loss.item()
            pred=torch.argmax(output,dim=1)
            correct+=(pred == Y).sum().item()
            total+=Y.size(0)
    avg_val_loss=total_val_loss/len(val_loader)
    val_acc=correct/total
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)
    print(
        f"Epoch{epoch+1}|"
        f"Train Loss:{avg_train_loss:.4f}|"
        f"Val Loss:{avg_val_loss:.4f}|"
        f"Val Acc:{val_acc:.4f}"
    )
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(),"best_model.pth")
        print("Saved best model")
df=pd.DataFrame({
    "epoch":list(range(1,config.EPOCHS+1)),
    "train_loss":train_losses,
    "val_loss":val_losses,
    "val_acc":val_accs,
    "lr":[config.LR]*config.EPOCHS,
    "batch_size":[config.BATCH_SIZE]*config.EPOCHS,
    "experiment_name":[config.EXPERIMENT_NAME]*config.EPOCHS
    })
df.to_csv(f"{config.EXPERIMENT_NAME}.csv",index=False)
print(f"Experiment saved as{config.EXPERIMENT_NAME}.csv")
    

