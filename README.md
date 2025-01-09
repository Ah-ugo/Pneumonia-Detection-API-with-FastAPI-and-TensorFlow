### Pneumonia Detection API with FastAPI and TensorFlow  

This repository contains an API for detecting pneumonia from chest X-ray images. Built with **FastAPI** and **TensorFlow**, it provides endpoints for real-time predictions and is designed to classify images into `NORMAL` or `PNEUMONIA` categories using a trained Convolutional Neural Network (CNN).  

---

### Features  
- **RESTful API**: Endpoints for uploading images and receiving predictions.  
- **Custom CNN Model**: Trains on chest X-ray datasets with data augmentation to reduce overfitting.  
- **Real-Time Prediction**: Accepts grayscale chest X-ray images and returns the predicted class with confidence scores.  
- **Handles Class Imbalance**: Incorporates class weights to improve model performance on imbalanced datasets.  
- **Startup Model Training**: Automatically trains the model on server startup if not already trained.  

---

### Installation  

1. **Clone the Repository**  
   ```bash  
   git clone https://github.com/Ah-ugo/Pneumonia-Detection-API-with-FastAPI-and-TensorFlow.git  
   cd Pneumonia-Detection-API-with-FastAPI-and-TensorFlow  
   ```  

2. **Install Dependencies**  
   Ensure you have Python 3.8 or higher installed, then run:  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. **Prepare Dataset**  
   Organize your dataset into the following structure:  
   ```  
   train/  
       NORMAL/  
       PNEUMONIA/  
   val/  
       NORMAL/  
       PNEUMONIA/  
   test/  
       NORMAL/  
       PNEUMONIA/  
   ```  

4. **Run the Server**  
   Start the FastAPI server:  
   ```bash  
   uvicorn main:app --reload  
   ```  
   Access the API documentation at `http://127.0.0.1:8000/docs`.  

---

### Endpoints  

1. **`GET /`**  
   - Returns a welcome message for the API.  

2. **`POST /predict`**  
   - Accepts a grayscale chest X-ray image as a file upload.  
   - Returns the predicted class (`NORMAL` or `PNEUMONIA`) and confidence score.  

---

### Example Usage  

**cURL Example**:  
```bash  
curl -X POST "http://127.0.0.1:8000/predict" \
-F "image=@path_to_your_image.jpeg"  
```  

**Sample Response**:  
```json  
{  
  "predicted_class": "PNEUMONIA",  
  "confidence": 0.92  
}  
```  

---

### Technologies Used  

- **FastAPI**: Framework for building RESTful APIs.  
- **TensorFlow**: Framework for machine learning and model training.  
- **OpenCV**: For image preprocessing.  
- **Keras Preprocessing**: For data augmentation.  

---

### Author  

Developed by **Ahuekwe Prince Ugochukwu** as part of a final year project at Abia State University.  
Based on the original work by [Flemming Kondrup](https://github.com/FlemmingKondrup/PneumoniaDiagnosisML).  

---

Feel free to contribute or raise issues to improve this repository! 😊