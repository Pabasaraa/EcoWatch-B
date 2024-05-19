from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from services.wildlife_comp.wildlife_service import WildlifeService
import uuid

router = APIRouter()
service = WildlifeService()

@router.get("/")
async def health_check():
    return {"component": "Wild Species Identification", "status": "ok"}

@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...), 
    location: str = Form(...), 
    date: str = Form(...), 
    time: str = Form(...),
    R_width_padding: int = Form(...), 
    L_width_padding: int = Form(...), 
    U_height_padding: int = Form(...), 
    D_height_padding: int = Form(...)
):
    try:
        with open(file.filename, "wb") as buffer:
            buffer.write(await file.read())
        
        output_video_path = file.filename.rsplit('.', 1)[0] + '_processed.mp4'
        
        result = service.process_video(file.filename, output_video_path, location, date, time, R_width_padding, L_width_padding, U_height_padding, D_height_padding)
        return {"message": "Video processing completed successfully.", "result": result}
    except Exception as e:
        return {"error": str(e)}


@router.get("/results/{task_id}")
async def get_results(task_id: uuid.UUID):
    result = service.get_results(task_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return {"Task_ID": task_id, "Results": result}