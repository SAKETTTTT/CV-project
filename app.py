import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import base64

from model import FacialExpressionModel

app = Flask(__name__)

model = FacialExpressionModel("model.json", "model_weights.h5")
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    # Get the JSON data from the request
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    
    # Decode the base64 image data to a numpy array
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_frame, 1.3, 5)

    # Process each face detected
    for (x, y, w, h) in faces:
        roi = gray_frame[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        final_roi = np.expand_dims(np.expand_dims(roi, axis=-1), axis=0)
        
        # Get emotion prediction
        prediction = model.predict_emotion(final_roi)

        # Draw a circle and text on the original color frame
        cv2.putText(frame, str(prediction), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 105, 255), 2)
        cv2.circle(frame, (int((x + x+w)/2), int((y + y+h)/2)), int(w/2), (0, 255, 0), 2)

    # Encode the processed frame back to base64 for display
    _, jpeg = cv2.imencode('.jpg', frame)
    processed_image_data = "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode('utf-8')

    # Return the processed image and emotion as JSON
    return jsonify({
        'processed_image': processed_image_data,
        'emotion': prediction
    })

if __name__ == '__main__':
    app.run(debug=True)
