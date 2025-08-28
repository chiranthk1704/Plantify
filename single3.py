import time
import warnings
import logging
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import psutil
import cv2
from contextlib import contextmanager, redirect_stdout
from ultralytics import YOLO
from pathlib import Path

warnings.simplefilter("ignore")
logging.getLogger('ultralytics').setLevel(logging.ERROR)

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

leaf_classes_1 = [
    'Apple', 'Apple', 'Apple', 'Apple', 'Blueberry', 'Cherry', 'Cherry',
    'Corn', 'Corn', 'Corn', 'Corn', 'Grape', 'Grape', 'Grape', 'Grape',
    'Orange', 'Peach', 'Peach', 'Pepper', 'Pepper', 'Potato', 'Potato',
    'Potato', 'Raspberry', 'Soybean', 'Squash', 'Strawberry', 'Strawberry',
    'Tomato', 'Tomato', 'Tomato', 'Tomato', 'Tomato', 'Tomato', 'Tomato',
    'Tomato', 'Tomato', 'Tomato'
]

leaf_model_path = 'plant_model.pth'
leaf_model = ResNet9(3, len(leaf_classes_1))
leaf_model.load_state_dict(torch.load(leaf_model_path, map_location=torch.device('cpu')))
leaf_model.eval()

def predict_resnet9(image_path, model=leaf_model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    try:
        initial_memory = get_memory_usage()
        start_time = time.time()
        with open(image_path, 'rb') as f:
            image = Image.open(io.BytesIO(f.read()))
            img_t = transform(image)
            img_u = torch.unsqueeze(img_t, 0)

        yb = model(img_u)
        probs = torch.nn.functional.softmax(yb, dim=1)
        confidence, preds = torch.max(probs, dim=1)
        time_taken = time.time() - start_time
        memory_usage = get_memory_usage() - initial_memory

        prediction = leaf_classes_1[preds[0].item()]
        confidence_score = round(confidence[0].item() * 100, 2)

        return [str(prediction), str(confidence_score), str(round(time_taken, 2)), str(round(memory_usage, 2))]
    except Exception as e:
        return [str(f"Error: {str(e)}"), "0.0", "0.0", "0.0"]

def processing_result_box(result):
    try:
        result_string = str(result)
        cleaned_result = result_string[7:-1]
        nested_list = json.loads(cleaned_result)
        class_names = [
            'ginger', 'banana', 'tobacco', 'ornamental', 'rose', 'soyabean',
            'papaya', 'garlic', 'raspberry', 'mango', 'cotton', 'corn', 'pomegranate',
            'strawberry', 'blueberry', 'brinjal', 'potato', 'wheat', 'olive',
            'rice', 'lemon', 'cabbage', 'guava', 'chilli', 'capsicum',
            'sunflower', 'cherry', 'cassava', 'apple', 'tea', 'sugarcane',
            'groundnut', 'weed', 'peach', 'coffee', 'cauliflower', 'tomato',
            'onion', 'gram', 'chiku', 'jamun', 'castor', 'pea', 'cucumber',
            'grape', 'cardamom'
        ]
        name_of_plant = class_names[int(nested_list[0][5])]
        confidence = round(nested_list[0][4] * 100, 2)
        return [str(name_of_plant), str(confidence)]
    except (json.JSONDecodeError, IndexError):
        return ['Unknown Plant', "0.0"]

yolo_model = YOLO('best.pt') 

def predict_yolo(image_path):
    try:
        initial_memory = get_memory_usage()
        start_time = time.time()
        image = cv2.imread(image_path)
        if image is None:
            return ["Error: Image not found or invalid format", "0.0", "0.0", "0.0"]

        results = yolo_model(image)
        time_taken = time.time() - start_time
        ans = processing_result_box(results[0].boxes)
        memory_usage = get_memory_usage() - initial_memory

        return [ans[0], ans[1], str(round(time_taken, 2)), str(round(memory_usage, 2))]
    except Exception as e:
        return [str(f"Error: {str(e)}"), "0.0", "0.0", "0.0"]

if __name__ == "__main__":
    image_path = '2_tomato.png'
    if Path(image_path).is_file():
        resnet_result = predict_resnet9(image_path)
        print("ResNet-9 Result:", resnet_result)
        yolo_result = predict_yolo(image_path)
        print("YOLO Result:", yolo_result)
    else:
        print(f"Error: Image file '{image_path}' not found.")
