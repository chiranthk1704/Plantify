import os
import sys
import warnings
import logging
import time
import psutil
from contextlib import contextmanager
from ultralytics import YOLO
import cv2
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger('ultralytics').setLevel(logging.ERROR)

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  

def predict_image(path):
    try:
        initial_memory = get_memory_usage()
        
        start_time = time.time()
        with suppress_stdout():
            model = YOLO('best.pt') 
        loading_time = time.time() - start_time  
        
        image = cv2.imread(path)
        if image is None:
            return ["Error: Image not found or invalid format", 0.0, 0.0, 0.0]
        
        start_time = time.time()
        with suppress_stdout():
            results = model(image)
        time_taken = time.time() - start_time  
        
        ans = processing_result_box(results[0].boxes)

        final_memory = get_memory_usage()
        memory_usage = final_memory - initial_memory
        
        return [ans[0], ans[1], time_taken, memory_usage]
    except Exception as e:
        return [f"Error: {str(e)}", 0.0, 0.0, 0.0]

def processing_result_box(result):
    """
    Process the YOLO result to extract class name and confidence score.
    :param result: YOLO model's result boxes
    :return: List [class_name, confidence]
    """
    try:
        uncleaned_result_string = str(result)
        cleaned_result_string = uncleaned_result_string[7:-1]
        nested_list = json.loads(cleaned_result_string)
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
        confidence = nested_list[0][4]*100
        return [name_of_plant, confidence]
    except (json.JSONDecodeError, IndexError):
        return ['Plant unable to be recognized', 0.0000]

if __name__ == "__main__":
    image_path = 'pepper.png'  
    if os.path.exists(image_path):
        result = predict_image(image_path)
        print(result)
    else:
        print(f"Error: Image file '{image_path}' not found.")
