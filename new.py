from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np

app = FastAPI()

# Load the pre-trained model
model = tf.keras.models.load_model("jarvish.h5")
CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary file
        img_path = 'temp_image.jpg'
        with open(img_path, 'wb') as f:
            f.write(file.file.read())

        # Preprocess the image
        img_array = preprocess_image(img_path)

        # Make a prediction
        predictions = model.predict(img_array)

        # Get the emotion label
        emotion_label = CLASS_LABELS[np.argmax(predictions)]

        # Return the result as JSON
        result = {'emotion': emotion_label}
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
