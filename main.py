from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

# Initialize FastAPI app
app = FastAPI(
    title="MNIST Digit Classifier API",
    version="1.0"
)

# Global model variable
model = None


# Load model at startup (SAFE WAY)
@app.on_event("startup")
def load_model_on_startup():
    global model
    try:
        model = load_model("mnist_model.h5")
        print("Model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")


# Health check
@app.get("/")
def read_root():
    return {
        "message": "MNIST Classification API is running. Send POST requests to /predict/"
    }


# Prediction endpoint
@app.post("/predict/")
async def predict_digit(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")

        # Resize to MNIST format
        image = image.resize((28, 28))

        # Convert to numpy
        image_array = np.array(image).astype("float32") / 255.0

        # Flatten for Dense model
        image_array = image_array.reshape(1, 784)

        # Predict
        prediction_probs = model.predict(image_array, verbose=0)
        predicted_class = int(np.argmax(prediction_probs))
        confidence = float(np.max(prediction_probs))

        return {
            "filename": file.filename,
            "predicted_digit": predicted_class,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")