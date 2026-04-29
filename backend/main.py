from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from logs.logger import logger
from dotenv import load_dotenv
from pathlib import Path
import asyncio  # ✅ fixed
import os, uuid, shutil, base64


from helpers.text_to_speech import text_to_speech
from gemini_llm.gemini_llm import client
from prompts.prompts import prompt_for_medical_image_analysis
import config as config
import uvicorn
from google.cloud import storage

BUCKET_NAME = "medai_bucket"
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)


load_dotenv()

app = FastAPI(title="Med AI backend application")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"]
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# IMAGE_DIR as Path
IMAGE_DIR = Path(config.IMAGE_DIR)
IMAGE_DIR.mkdir(exist_ok=True)

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/upload-image")
async def get_image_detail(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only images accepted")

    ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{ext}"

    try:
        blob = bucket.blob(filename)

        blob.upload_from_file(
            file.file,
            content_type=file.content_type
        )

        return {
            "status": "uploaded",
            "file_name": filename
        }

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(503, "Service unavailable")
    

@app.post("/analyze-image")
async def analyze_image(file_name: str):

    file_path = IMAGE_DIR / file_name
    blob = bucket.blob(file_name)

    if not blob.exists():
        raise HTTPException(404, "File not found in bucket")

    blob.download_to_filename(str(file_path))

    fetch_file = client.files.upload(
        file=str(file_path)
    )

    async def generate():
        try:
            for chunk in client.models.generate_content_stream(
                model="gemini-2.5-flash-lite",
                contents=[
                    fetch_file,
                    prompt_for_medical_image_analysis
                ]
            ):
                if chunk.text:
                    yield chunk.text
                    await asyncio.sleep(0)

        finally:
            # ✅ Delete Gemini file
            try:
                client.files.delete(name=fetch_file.name)
            except:
                pass

            # ✅ Delete from GCS
            try:
                blob.delete()
            except:
                pass

            # ✅ Delete local temp
            if file_path.exists():
                file_path.unlink()

    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)