import torch
import torch.nn as nn
from torchvision import transforms,datasets
from sklearn.metrics import confusion_matrix,precision_score,recall_score
import config
import utils
from torch.utils.data import DataLoader,random_split
from models.resnet_model import get_model
import torch.optim as optim
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
dataset=datasets.ImageFolder(config.DATA_PATH,transform=transform)
train_size=int(0.8*len(dataset))
val_size=len(dataset)-train_size
_,val_set=random_split(dataset,[train_size,val_size])
val_loader=DataLoader(val_set,batch_size=config.BATCH_SIZE)
model=get_model()
all_preds=[]
all_labels=[]
model.eval()
with torch.no_grad():
    for X,Y in val_loader:
        output=model(X)
        pred=torch.argmax(output,dim=1)
        all_preds.extend(pred.tolist())
        all_labels.extend(Y.tolist())
confusion_matrix_result=confusion_matrix(all_labels,all_preds)
precision=precision_score(all_labels,all_preds,zero_division=0)
recall=recall_score(all_labels,all_preds,zero_division=0)
print("cm",confusion_matrix_result)
print("precision_score",precision)
print("recall_score",recall)
utils.print_distribution(all_preds,all_labels)
for i in range(len(all_labels)):
    if all_preds[i] != all_labels[i]:
        print(f"錯誤index{i}:真實={all_labels[i]},預測={all_preds[i]}") 