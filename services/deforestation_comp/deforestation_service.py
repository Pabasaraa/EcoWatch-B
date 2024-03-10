from fastapi import BackgroundTasks, UploadFile, HTTPException
from PIL import Image
import asyncio
import uuid
import base64
import os
import io
from models.deforestation_prediction.model import predict
from models.deforestation_prediction.preprocess_lidar import laz2las, las2tif

# Placeholder for storing progress data and results
task_status = {}
prediction_results = {}


class DeforestationService():

    async def initiate_prediction(self, file: UploadFile, background_tasks: BackgroundTasks):
        name, extension = os.path.splitext(file.filename)

        if extension not in [".png", ".jpg", ".jpeg", ".tif", ".laz", ".las", ".lasd"]:
            raise HTTPException(
                status_code=400, detail="File type not supported")

        task_id = uuid.uuid4()
        background_tasks.add_task(self.predict_deforestation, file, task_id)

        return task_id

    async def predict_deforestation(self, file: UploadFile, task_id: uuid.UUID):
        name, extension = os.path.splitext(file.filename)
        contents = await file.read()

        if extension in [".laz"]:
            las_file = await self.__convert_laz_to_las(task_id, name, contents)
            if las_file:
                tif_file = await self.__convert_las_to_tif(task_id, name, las_file)
                if tif_file:
                    prediction = await self.__process_model(task_id, name, tif_file)

        elif extension in [".las", ".lasd"]:
            tif_file = await self.__convert_las_to_tif(task_id, name, contents)
            if tif_file:
                prediction = await self.__process_model(task_id, name, tif_file)

        else:
            image = Image.open(io.BytesIO(contents))
            prediction, value, input_img = await self.__process_model(task_id, name, image)

        prediction_results[task_id] = {"filename": file.filename,
                                       "prediction": prediction,
                                       "value": value,
                                       "input_img": input_img}

    async def __convert_laz_to_las(self, task_id: uuid.UUID, name: str, contents: bytes):
        self.add_status(task_id, f'Converting {name} to LAS', False)
        await asyncio.sleep(1)
        las_file = laz2las(contents)
        if las_file:
            self.update_status(task_id, f'Converting {name} to LAS', True)
        else:
            raise Exception(f'Error converting {name} to LAS')

        return las_file

    async def __convert_las_to_tif(self, task_id: uuid.UUID, name: str, las_file):
        self.add_status(task_id, f'Converting {name} to TIF', False)
        await asyncio.sleep(1)
        tif_file = las2tif(las_file)
        if tif_file:
            self.update_status(task_id, f'Converting {name} to TIF', True)
        else:
            raise Exception(f'Error converting {name} to TIF')

        return tif_file

    async def __process_model(self, task_id: uuid.UUID, name: str, input_img):
        self.add_status(task_id, f"Detecting deforestation in {name}", False)
        await asyncio.sleep(1)
        if input_img:
            prediction, value = predict(input_img)
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

        if prediction_base64 is not None:
            self.update_status(
                task_id, f"Detecting deforestation in {name}", True)
        else:
            raise Exception(f'Error processing {name}')

        return prediction_base64, value, input_img_base64

    def add_status(self, task_id, status, completed):
        if task_id not in task_status:
            task_status[task_id] = []
        task_status[task_id].append({'status': status, 'completed': completed})

    def update_status(self, task_id, status, completed):
        if task_id in task_status:
            for task in task_status[task_id]:
                if task['status'] == status:
                    task['completed'] = completed
                    break

    def get_progress(self, task_id: uuid.UUID):
        progress = task_status.get(task_id, None)
        return progress

    def get_results(self, task_id: uuid.UUID):
        results = prediction_results.get(task_id, None)
        return results
