# Let us import the Libraries required.
import os
import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# Importing the required Classes/Functions from Modules defined.
from model import FacialExpressionModel

# Let us Instantiate the app
app = Flask(__name__)
# Initialize Flask-SocketIO
socketio = SocketIO(app)

# Load your pre-trained model and face cascade classifier
model = FacialExpressionModel("model.json", "model_weights.h5")
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

###################################################################################

@app.route('/')
def index():
    """ Renders the real-time video streaming home page. """
    return render_template('index.html')


@socketio.on('video_frame')
def handle_video_frame(data):
    """
    Handles incoming video frames from the client via WebSocket.
    Processes the frame and sends back the result.
    """
    # Decode the image data sent from the client
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    # Convert the frame to grayscale for face detection and emotion recognition
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

    # Encode the processed frame back to JPEG format
    _, jpeg = cv2.imencode('.jpg', frame)
    
    # Send the processed frame back to the client
    emit('processed_frame', jpeg.tobytes())


if __name__ == '__main__':
    # Use SocketIO's run function to start the server
    socketio.run(app, debug=True)