import torch
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO('best.pt')

# Export to ONNX
model.export(format='onnx', simplify=True, opset=12)