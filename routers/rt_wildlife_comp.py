from fastapi import APIRouter, UploadFile, File, HTTPException
from services.wildlife_comp.wildlife_service import WildlifeService
import os

router = APIRouter()
service = WildlifeService()

@router.get("/")
async def health_check():
    return {"component": "Wild Species Identification", "status": "ok"}


@router.post("/process_video")
async def process_video(file: UploadFile = File(...)):
    try:
        video_dir = 'uploaded_videos'
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        
        video_path = os.path.join(video_dir, file.filename)
        with open(video_path, "wb") as buffer:
            buffer.write(await file.read())
        service.process_video(video_path)
        return {"message": "Video processing completed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
