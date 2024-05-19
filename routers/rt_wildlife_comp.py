from fastapi import APIRouter, UploadFile, File, HTTPException
from services.wildlife_comp.wildlife_service import WildLifeService
import os
import uuid

router = APIRouter()
service = WildLifeService()

@router.get("/")
async def health_check():
    return {"component": "Wild Species Identification", "status": "ok"}


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        with open(file.filename, "wb") as buffer:
            buffer.write(await file.read())
        
        output_video_path = file.filename.rsplit('.', 1)[0] + '_processed.mp4'
        result = service.process_video(file.filename, output_video_path)
        return {"message": "Video processing completed successfully.", "result": result}
    
        # result = service.process_video(file.filename)
        # return {"task_id": task_id}
    except Exception as e:
        return {"error": str(e)}


@router.get("/results/{task_id}")
async def get_results(task_id: uuid.UUID):
    result = service.get_results(task_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return {"Task_ID": task_id, "Results": result}