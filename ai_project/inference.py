import torch
from torchvision import transforms
from PIL import Image
from models.resnet_model import get_model
class_name=["cat","dog"]
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
model=get_model(num_classes=2)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
def predict(image_pth):
    image=Image.open(image_pth).convert("RGB")
    image=transform(image).unsqueeze(0)
    with torch.no_grad():
        output=model(image)
        pred=torch.argmax(output,dim=1).item()
    return class_name[pred]
if __name__ == "__main__":
    image_pth="test.jpg"
    result=predict(image_pth)
    print("Prediction",result)
