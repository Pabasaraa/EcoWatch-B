from fastapi import APIRouter, UploadFile, File, HTTPException
from services.wildlife_comp.wildlife_service import process_video
import os

router = APIRouter()

@router.get("/")
async def health_check():
    return {"component": "Wild Species Identification", "status": "ok"}


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        with open(file.filename, "wb") as buffer:
            buffer.write(await file.read())
        
        output_video_path = file.filename.rsplit('.', 1)[0] + '_processed.mp4'
        
        result = process_video(file.filename, output_video_path)
        return {"message": "Video processing completed successfully.", "result": result}
    except Exception as e:
        return {"error": str(e)}