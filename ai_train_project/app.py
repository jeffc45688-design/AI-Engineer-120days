import io 
import time
from datetime import datetime
import torch
import torch.nn.functional as F
from torchvision import transforms
from models.resnet_model import get_model
from PIL import Image,UnidentifiedImageError
from fastapi import FastAPI,File,UploadFile
app=FastAPI()
MODEL_VERSION="v1.0"
class_name=["cat","dog"]
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
model=get_model()
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
@app.get("/")
def home():
    return {"message":"AI dog/cat classifier API is running"}
@app.post("/prediction")
async def prediction(file:UploadFile=File(...)):
    allow_file_types=["image/jpeg","image/png"]
    if file.content_type not in allow_file_types:
        return {"error":"Invalid file type.Please upload the JPG or PNG image"}
    image_bytes=await file.read()
    if len(image_bytes) == 0:
        return {"error":"Invalid file.Please upload the image"}
    try:
        image=Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image=transform(image).unsqueeze(0)
        start_time=time.time()
        with torch.no_grad():
            output=model(image)
            probs=F.softmax(output,dim=1)
            values,indices=torch.topk(probs,k=2,dim=1)
        end_time=time.time()
        inference_time=round(end_time-start_time,4)
        results=[]
        for i in range(len(indices[0])):
            results.append({
                "label":class_name[indices[0][i].item()],
                "confidence":round(values[0][i].item(),4)
            })

        return {
            "filename":file.filename,
            "model_version":MODEL_VERSION,
            "inference_time":inference_time,
            "timestamp":str(datetime.now()),
            "top2_prediction":results
        }
    except UnidentifiedImageError:
        return {"error":"Invaild file"}
    except Exception as e:
        return {"error":f"Unexcepted sever error{str(e)}"}
