
# Textile Defect Detection

Installation
To install the required dependencies, run the following command:

```
pip install -r requirements.txt
```

Usage
To start the Flask server, run the following command:


```python app.py
You can then access the application at http://localhost:8000.
```

To test the application, you can upload an image of a cloth with watermarks and click on the "Submit" button. The application will use YOLOv8 to detect the watermarks and highlight them in the image.

Dataset
The dataset used to train the YOLOv8 model consists of images of cloth on a conveyor belt with watermarks. The dataset is included in this repository.

Custom Model
The YOLOv8 model used in this application was trained on the cloth dataset. The model is not included in this repository,but it has been uploaded to [![Google Drive](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Google_Drive_text_logo_grey.png/100px-Google_Drive_text_logo_grey.png)](https://drive.google.com/file/d/1uEna3lt5sW4v_LeBQ2Oi1MhMyz5f5flG/view?usp=sharing) as CustomModelWater.pt.

Results
Here are some sample images from the training and validation batches, as well as a sample result from the application:

runs/segment/yolov8n_custom/result.png: Result
runs/segment/yolov8n_custom/val_batch0_pred.jpg: Sample Predictions



License
This application is licensed under the MIT License. See the LICENSE file for more information.
