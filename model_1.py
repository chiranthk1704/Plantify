import torch
from torchvision import transforms
from PIL import Image
import io
import os
import sys
import time
import warnings
import psutil

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

from model import ResNet9

leaf_classes = [
    'Apple', 'Apple', 'Apple', 'Apple', 'Blueberry', 'Cherry', 'Cherry',
    'Corn', 'Corn', 'Corn', 'Corn', 'Grape', 'Grape', 'Grape', 'Grape',
    'Orange', 'Peach', 'Peach', 'Pepper', 'Pepper', 'Potato', 'Potato',
    'Potato', 'Raspberry', 'Soybean', 'Squash', 'Strawberry', 'Strawberry',
    'Tomato', 'Tomato', 'Tomato', 'Tomato', 'Tomato', 'Tomato', 'Tomato',
    'Tomato', 'Tomato', 'Tomato'
]

leaf_model_path = 'plant_model.pth'  
leaf_model = ResNet9(3, len(leaf_classes))
leaf_model.load_state_dict(torch.load(leaf_model_path, map_location=torch.device('cpu'), weights_only=True))
leaf_model.eval()

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) 
def predict_image(image_path, model=leaf_model):
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
        end_time = time.time()
        time_taken = end_time - start_time 
        final_memory = get_memory_usage()
        memory_usage = final_memory - initial_memory 
        
        prediction = leaf_classes[preds[0].item()]
        confidence_score = confidence[0].item() * 100  
        
        return [prediction, confidence_score, time_taken, memory_usage]
    except Exception as e:
        return f"Error processing the image: {str(e)}"

if __name__ == "__main__":
    image_path = 'pepper.png'  
    if os.path.exists(image_path):
        result = predict_image(image_path)
        if isinstance(result, list):
            print(result)
    else:
        print(f"Error: Image file '{image_path}' not found.")
