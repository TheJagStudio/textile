from flask import Flask, Response, send_file, make_response, request, render_template, redirect, url_for, flash, session
from ultralytics import YOLO
import numpy as np
import cv2
import json

app = Flask(__name__)
model = YOLO("CustomModelWater.pt")

@app.route('/demo', methods = ['POST'])
def demo():
    if request.method == 'POST':  
        inputImage = request.files['image']
        img = cv2.imdecode(np.frombuffer(inputImage.read(), np.uint8), cv2.IMREAD_COLOR)
        results = model.predict(source=img ,show_labels=False,conf=0.1,boxes=False)
        image = results[0].plot()
        _, img_encoded = cv2.imencode('.png', image)
        response = make_response(img_encoded.tobytes())
        response.headers['Content-Type'] = 'image/png'
        return response
    else:
        return Response("Method not allowed", status=405)


@app.route('/')
def hello_world():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port="8000")