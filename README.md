# Plantify: Plant Recognition with ResNet and YOLO

## Technologies and Tools Used

- **Android Studio (Kotlin)** – to build the mobile app.  
- **Python (TensorFlow/Keras, PyTorch)** – to create and test the models.  
- **ResNet, Sequential CNN, and YOLO** – deep learning models used for comparison.  
- **SQLite** – used inside the app for storing results.  

---

## Steps to Run

1. Download or clone this repository and open it in **Android Studio**.  
2. Let Gradle finish syncing the dependencies.  
3. Build and run the project on an Android emulator or a connected phone.  
4. The app already has the models integrated, so it can run directly.  

---

## Features of the App

- **Compare Models**  
  The app lets you see the difference between a simple Sequential CNN, ResNet, and YOLO for plant recognition.  

- **Vanishing Gradient Issue**  
  Shows how deeper CNNs struggle with vanishing gradients and how ResNet improves training with residual connections.  

- **Performance Display**  
  For each test, the app shows:  
  - Time taken for the prediction  
  - Memory usage  
  - Output of the model (predicted plant type/condition)  

- **History**  
  Results are stored in a local database so that past tests can be checked later.  

---

## Learning Outcomes

This project helped in:  
- Understanding how residual connections make training deeper networks easier.  
- Seeing the difference in speed, memory, and accuracy between Sequential CNN, ResNet, and YOLO.  
- Learning how to bring deep learning models into an Android app.  

---

## Demo Images

(Add screenshots here)
