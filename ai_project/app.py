import io
import torch
import time
from datetime import datetime
from models.resnet_model import get_model
from fastapi import FastAPI,File,UploadFile
from torchvision import transforms
from PIL import Image,UnidentifiedImageError
import torch.nn.functional as F
app=FastAPI()
class_name=["cat","dog"]
MODEL_VERSION="v1.0"
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
model=get_model()
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
@app.get("/")
def home():
    return {"massage":"AI dog/cat classifier API is running"}
@app.post("/prddiction")
async def prediction(file:UploadFile=File(...)):
    allow_file_type=["image/jpeg","image/png"]
    if file.content_type not in allow_file_type:
        return {"error":"Unvalid file type.Plezse upload  the JPG or PNG image"}
    image_bytes=await file.read()
    if len(image_bytes) == 0:
        return {"error":"Invalid file"}
    try:
        image=Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image=transform(image).unsqueeze(0)

        start_time=time.time()
    
        with torch.no_grad():
            output=model(image)
            probs=F.softmax(output,dim=1)  #argmax 類別(索引) max 類別+機率 因為有confidence
            values,indices=torch.topk(probs,k=2,dim=1) #values機率 indices類別位置
       
        end_time=time.time()
        inference_time=round(end_time-start_time,4)
    
        results=[]                                           # topk 取前k個最大值及"位置"
        for i in range(len(indices[0])):
            results.append({
                "labels":class_name[indices[0][i].item()],
                "confidence":round(values[0][i].item(),4)
            })

        return {
            "filename":file.filename,
            "model_version":MODEL_VERSION,
            "inference_time":inference_time,
            "timestamp":str(datetime.now()),
            "top3_prediction":results
        }
    except UnidentifiedImageError:
        return {"error":"Upload file is not a valid file"}
    except Exception as e:
        return {"error":f"Unexcepted sever error{str(e)}"}
#執行 uvicorn app:app --reload
#/docs
#indices是表示哪個索引(類別)大
#time.time() 秒數(float) 用途:計算時間
#datatime.now() 日期時間 用途:顯示時間

