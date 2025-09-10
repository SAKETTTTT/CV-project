# Let us import the Libraries required.
import cv2
import numpy as np

from model import FacialExpressionModel

# Creating an instance of the class with the parameters as model and its weights.
model = FacialExpressionModel("model.json", "model_weights.h5")

# Loading the classifier from the file.
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class VideoCamera(object):
    """ Takes the Real time Video, Predicts the Emotion using pre-trained model. """

    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        """It returns camera frames along with bounding boxes and predictions"""
        _, frame = self.video.read()

        if frame is None:
            # Handle case where no frame is captured (e.g., camera not available)
            return None

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scaleFactor = 1.3
        minNeighbors = 5
        faces = facec.detectMultiScale(gray_frame, scaleFactor, minNeighbors)

        for (x, y, w, h) in faces:
            roi = gray_frame[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))

            # The correct reshaping for a single grayscale image
            # Shape should be (1, 48, 48, 1) -> (batch_size, height, width, channels)
            final_roi = np.expand_dims(np.expand_dims(roi, axis=-1), axis=0)
            
            prediction = model.predict_emotion(final_roi)

            Text = str(prediction)
            Text_Color = (180, 105, 255)
            Thickness = 2
            Font_Scale = 1
            Font_Type = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, Text, (x, y), Font_Type, Font_Scale, Text_Color, Thickness)

            xc = int((x + x+w)/2)
            yc = int((y + y+h)/2)
            radius = int(w/2)
            cv2.circle(frame, (xc, yc), radius, (0, 255, 0), Thickness)

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
