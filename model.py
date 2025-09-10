import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json, Sequential

from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class YourSequential(Sequential):
    pass

class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
        # Use custom_objects to map 'Sequential' to your registered class
        self.loaded_model = model_from_json(loaded_model_json, custom_objects={'Sequential': YourSequential})
        self.loaded_model.load_weights(model_weights_file)

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]

    def return_probabs(self, img):
        self.preds = self.loaded_model.predict(img)
        return self.preds
