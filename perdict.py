import torch
from PIL import Image
import json
import time
import psutil
import cv2
from ultralytics import YOLO
from torchvision import transforms

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def predict_yolo(image_path):
    model = YOLO('best.pt')  
    image = cv2.imread(image_path)
    results = model(image)
    class_names = ['ginger', 'banana', 'tomato', 'apple', 'cherry', 'orange']  
    prediction = str(class_names[int(results[0].boxes[0][5])]) 
    confidence = round(results[0].boxes[0][4] * 100, 2)
    return prediction, confidence

def predict_resnet9(image_path):
    transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
    image = Image.open(image_path)
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    model = torch.load('resnet9_model.pth')  
    model.eval()
    with torch.no_grad():
        yb = model(img_u)
        probs = torch.nn.functional.softmax(yb, dim=1)
        confidence, preds = torch.max(probs, dim=1)
        prediction = preds[0].item()  
        confidence_score = round(confidence[0].item() * 100, 2)
        return prediction, confidence_score

if __name__ == "__main__":
    image_path = 'a_corn.png'  
    resnet_result = predict_resnet9(image_path)
    yolo_result = predict_yolo(image_path)

    print(f"ResNet9 Result: {resnet_result}")
    print(f"YOLO Result: {yolo_result}")
