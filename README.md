# Face Emotion Detector

A small Flask web app that looks at a face through your webcam and predicts the emotion. Press a button, the browser sends a snapshot to the server, and you get the photo back with the face outlined and the predicted emotion drawn on it.

It recognises seven emotions: **Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise**.

## How it works

1. The browser opens the webcam (`getUserMedia`) and sends one captured frame to `POST /analyze_image`.
2. **OpenCV** converts the frame to grayscale and finds faces with a Haar cascade (`haarcascade_frontalface_default.xml`).
3. Each face is cropped and resized to 48×48 pixels and passed to a **Keras convolutional neural network** (4 convolution blocks with batch normalization, max pooling and dropout, followed by dense layers).
4. The server draws the result on the frame and returns it with the predicted emotion as JSON.

If no face is found, the app answers "No face detected".

The network was trained in `Facial Expression Recognition.ipynb`; the saved architecture (`model.json`) and trained weights (`model_weights.h5`) are included, so you can run the app without training anything.

## Tech stack

Python 3.11 · Flask · TensorFlow/Keras 2.15 · OpenCV · NumPy

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000, allow camera access, and click **Capture Image and Analyze Emotion**.

Debug mode is off by default. To turn it on while developing: `FLASK_DEBUG=1 python app.py`.

## Project layout

| Path | What it is |
|---|---|
| `app.py` | Flask app: serves the page and the `/analyze_image` endpoint |
| `model.py` | Loads the saved model and returns the emotion label for a face |
| `model.json`, `model_weights.h5` | Trained network (architecture and weights) |
| `haarcascade_frontalface_default.xml` | OpenCV face detector |
| `Facial Expression Recognition.ipynb` | Notebook used to build and train the model |
| `templates/index.html`, `static/style.css` | Web page |
| `camera.py`, `Graphical_Visualisation.py`, other templates | From an earlier version of the app; not used by `app.py` |
| `tests/` | pytest fixtures (`pytest` needs the dependencies above installed) |

## Limits

- It only classifies the single facial expression in the photo, so accuracy depends on lighting, angle and image quality.
- Emotion labels from a model like this are a rough guess, not a reliable read of how someone actually feels.
- Images are processed in memory and are not stored on the server.
