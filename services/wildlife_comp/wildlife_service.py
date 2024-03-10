from fastapi import BackgroundTasks, UploadFile, HTTPException
from PIL import Image
import asyncio
import uuid
import base64
import os
import io
from models.wildlife_identification.model import predict


prediction_results = {}


class WildlifeService():

    async def initiate_prediction(self, file: UploadFile):
        name, extension = os.path.splitext(file.filename)
        contents = await file.read()

        if extension not in [".png", ".jpg", ".jpeg", ".tif"]:
            raise HTTPException(
                status_code=400, detail="File type not supported")

        task_id = uuid.uuid4()

        image = Image.open(io.BytesIO(contents))
        prediction, count, input_img = await self.__process_model(name, image)

        prediction_results[task_id] = {"filename": file.filename,
                                        "prediction": prediction,
                                        "count": count,
                                        "input_img": input_img}
        
        return task_id
        
        
    async def __process_model(self, name: str, input_img):
        if input_img:
            prediction, count = predict(input_img)
            normalized_prediction = (
                prediction - prediction.min()) / (prediction.max() - prediction.min()) * 255
            prediction_image = Image.fromarray(
                normalized_prediction.astype('uint8'))
            buffered = io.BytesIO()
            prediction_image.save(buffered, format="JPEG")
            prediction_base64 = base64.b64encode(buffered.getvalue()).decode()

            buffered_input = io.BytesIO()
            input_img.save(buffered_input, format="JPEG")
            input_img_base64 = base64.b64encode(
                buffered_input.getvalue()).decode()

        if prediction_base64 is None:
            raise Exception(f'Error processing {name}')

        return prediction_base64, count, input_img_base64
    
    

    def get_results(self, task_id:uuid.UUID):
        results = prediction_results.get(task_id, None)
        return results
