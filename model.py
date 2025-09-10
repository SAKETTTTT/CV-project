import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import register_keras_serializable

# Decorate for serializable fix if needed
@register_keras_serializable()
class YourSequential(Sequential):
    pass

class FacialExpressionModel(object):

    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()

        # Use custom_objects mapping for 'Sequential'
        self.loaded_model = model_from_json(
            loaded_model_json,
            custom_objects={'Sequential': YourSequential}
        )
        
        # If your model JSON does not include Input layer, rebuild the model with Input layer explicitly:
        # Example (optional, if needed):
        # input_layer = Input(shape=(48, 48, 1))
        # self.loaded_model.build(input_shape=(None, 48, 48, 1))

        self.loaded_model.load_weights(model_weights_file)

    def predict_emotion(self, img):
        # Ensure img shape is (1, 48, 48, 1)
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]

    def return_probabs(self, img):
        self.preds = self.loaded_model.predict(img)
        return self.preds
