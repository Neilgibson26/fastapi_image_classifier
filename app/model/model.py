import pickle
from pathlib import Path
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
from io import BytesIO


__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent



with open(f"{BASE_DIR}/trash_model.pkl", "rb") as f:
    model = pickle.load(f)


classes = [
    "Recyclable",
    "Non_Recyclable",
    "Compost"
]

def read_image(file) -> Image.Image:
    pil_image = Image.open(BytesIO(file))
    return pil_image

def transformacao(file: Image.Image):

    img = np.asarray(file.resize((128, 128)))[..., :3]
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)

    print('Predicted:', preds)
    
    
    if preds[0][1] == 1:
        return("Recyclable")
    elif preds[0][0] == 1:
        return("Not Recyclable")
    elif preds[0][2] == 1:
        return("Compost Bin")



