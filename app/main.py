from PIL import Image
from pydantic import BaseModel
from model.model import read_image
from model.model import transformacao
from fastapi import FastAPI, File, UploadFile
from io import BytesIO



app = FastAPI()


class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    trash_label: str



@app.get("/")
def home():
    return {"Hello": "World"}


@app.post("/predict", response_model=PredictionOut)
async def create_upload_file(file: bytes = File(...)):

    # read image
    imagem = read_image(file)
    # transform and prediction 
    prediction = transformacao(imagem)

    return {"trash_label" : prediction}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000) 