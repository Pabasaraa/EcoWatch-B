from fastapi import APIRouter, WebSocket, UploadFile, BackgroundTasks, Depends, File, HTTPException
from websockets.exceptions import ConnectionClosed
import uuid
import asyncio
import copy
from services.wildlife_comp.wildlife_service import WildlifeService

router = APIRouter()
service = WildlifeService()


@router.get("/")
async def health_check():
    return {"component": "Wild Species Identification", "status": "ok"}


@router.post("/predict")
async def detect_wildlife(file: UploadFile = File(...)):
    try:
        task_id = await service.initiate_prediction(file)
        return {"task_id": task_id}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{task_id}")
async def get_results(task_id: uuid.UUID):
    result = service.get_results(task_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return {"Task_ID": task_id, "Results": result}