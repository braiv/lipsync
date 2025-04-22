from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi
from typing import Optional
import os
import shutil
from pathlib import Path
import uuid
from facefusion.processors.modules.lip_syncer import process_video, process_image
from facefusion.filesystem import is_image, is_video
import facefusion.state_manager as state_manager
from pydantic import BaseModel


class TaskStatus(BaseModel):
    status: str
    progress: int
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    output_path: Optional[str] = None
    error: Optional[str] = None


class TaskResponse(BaseModel):
    status: str
    message: str
    task_id: str


app = FastAPI(
    title="Lipsync API",
    description="""
    A powerful API for lipsync processing that combines video and audio files to create synchronized output.
    
    ## Features
    * Upload video and audio files
    * Process lipsync in background
    * Track processing status
    * Download processed results
    
    ## Endpoints
    * `/lipsync/process` - Upload and process files
    * `/lipsync/status/{task_id}` - Check processing status
    
    ## Example
    ```bash
    curl -X POST "http://localhost:8000/lipsync/process" \\
      -H "accept: application/json" \\
      -H "Content-Type: multipart/form-data" \\
      -F "video=@input.mp4" \\
      -F "audio=@audio.mp3"
    ```
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Create upload and output directories if they don't exist
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Store task statuses
task_statuses = {}


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to the Lipsync API"}


@app.post(
    "/lipsync/process",
    response_model=TaskResponse,
    summary="Process lipsync",
    description="""
    Upload video and audio files for lipsync processing.
    
    The API will:
    1. Accept video and audio files
    2. Start processing in background
    3. Return a task ID for status tracking
    
    Supported video formats: MP4, MOV, AVI
    Supported audio formats: MP3, WAV, M4A
    """,
    responses={
        200: {
            "description": "Processing started successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "Processing started",
                        "task_id": "550e8400-e29b-41d4-a716-446655440000",
                    }
                }
            },
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {"example": {"detail": "Error processing files"}}
            },
        },
    },
)
async def process_lipsync(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Video file containing the face"),
    audio: UploadFile = File(..., description="Audio file to sync with the video"),
) -> TaskResponse:
    try:
        task_id = str(uuid.uuid4())

        video_path = UPLOAD_DIR / f"{task_id}_{video.filename}"
        audio_path = UPLOAD_DIR / f"{task_id}_{audio.filename}"

        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        task_statuses[task_id] = {
            "status": "processing",
            "progress": 0,
            "video_path": str(video_path),
            "audio_path": str(audio_path),
            "output_path": None,
        }

        background_tasks.add_task(process_task, task_id, video_path, audio_path)

        return TaskResponse(
            status="success", message="Processing started", task_id=task_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video:
            video.file.close()
        if audio:
            audio.file.close()


@app.get(
    "/lipsync/status/{task_id}",
    response_model=TaskStatus,
    summary="Get processing status",
    description="""
    Check the status of a lipsync processing task.
    
    Returns:
    * Current processing status
    * Progress percentage
    * Output file path (when complete)
    * Error message (if failed)
    """,
    responses={
        200: {
            "description": "Task status retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "processing",
                        "progress": 50,
                        "video_path": "/uploads/550e8400-e29b-41d4-a716-446655440000_video.mp4",
                        "audio_path": "/uploads/550e8400-e29b-41d4-a716-446655440000_audio.mp3",
                        "output_path": None,
                    }
                }
            },
        },
        404: {
            "description": "Task not found",
            "content": {"application/json": {"example": {"detail": "Task not found"}}},
        },
    },
)
async def get_lipsync_status(task_id: str) -> TaskStatus:
    if task_id not in task_statuses:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskStatus(**task_statuses[task_id])


async def process_task(task_id: str, video_path: Path, audio_path: Path):
    try:
        state_manager.set_item("source_paths", [str(audio_path)])
        state_manager.set_item("target_path", str(video_path))

        output_path = (
            OUTPUT_DIR / f"{task_id}_{video_path.stem}_output{video_path.suffix}"
        )
        state_manager.set_item("output_path", str(output_path))

        if is_video(str(video_path)):
            temp_frame_paths = [
                str(OUTPUT_DIR / f"{task_id}_frame_{i}.jpg") for i in range(100)
            ]
            process_video([str(audio_path)], temp_frame_paths)
        else:
            process_image([str(audio_path)], str(video_path), str(output_path))

        task_statuses[task_id].update(
            {"status": "completed", "progress": 100, "output_path": str(output_path)}
        )

    except Exception as e:
        task_statuses[task_id].update({"status": "failed", "error": str(e)})


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Lipsync API",
        version="1.0.0",
        description="""
        A powerful API for lipsync processing that combines video and audio files to create synchronized output.
        
        ## Features
        * Upload video and audio files
        * Process lipsync in background
        * Track processing status
        * Download processed results
        
        ## Example
        ```bash
        curl -X POST "http://localhost:8000/lipsync/process" \\
          -H "accept: application/json" \\
          -H "Content-Type: multipart/form-data" \\
          -F "video=@input.mp4" \\
          -F "audio=@audio.mp3"
        ```
        """,
        routes=app.routes,
    )

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
