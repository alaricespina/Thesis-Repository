from typing import Union, List, Dict
from fastapi import FastAPI
from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf

app = FastAPI()

# Load the model
model = tf.keras.models.load_model("iris_model.keras")

app = FastAPI()

# Optional CORS for local dev
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Request schema
class IrisInput(BaseModel):
    data: List[float]  # expects 4 features

@app.post("/predict/")
async def predict(input_data: IrisInput):
    input_array = np.array(input_data.data).reshape(1, -1)
    prediction = model.predict(input_array)
    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "class_index": class_index,
        "confidence": confidence,
        "class_name": ["setosa", "versicolor", "virginica"][class_index]
    }



# class Item(BaseModel):
#     name: str
#     tags: List[str]
#     values: List[float]


# @app.get("/")
# async def read_root():
#     return {"Hello": "World"}


# @app.get("/items/{item_id}")
# async def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}


# @app.post("/items/")
# async def create_item(item: Item):
#     return {
#         "message": "Item Received",
#         "item": item,
#         "hatdog": "hatdog"
#     }
