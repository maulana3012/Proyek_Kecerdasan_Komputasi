from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from typing import List
from sqlalchemy.orm import Session

from app.database import get_db, engine, Base
from app.es_client import init_es_indices
from app.models.student import StudentProfile
from app.services.face_recognizer import LBPHFaceRecognizerService

# Buat tabel MySQL otomatis jika belum ada
Base.metadata.create_all(bind=engine)

# Inisialisasi Elasticsearch Index
init_es_indices()

app = FastAPI(title="Face Recognition Microservice", version="3.0.0")
face_service = LBPHFaceRecognizerService()

@app.get("/")
def read_root():
    return {"status": "online", "service": "Face Recognition FastAPI"}

@app.post("/register")
async def register(
    nim: str = Form(...),
    nama: str = Form(...),
    prodi: str = Form(...),
    kelas: str = Form(...),
    photos: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    student = StudentProfile(nim=nim, nama=nama, prodi=prodi, kelas=kelas)
    images_bytes = [await photo.read() for photo in photos]

    saved_count = face_service.save_dataset_and_train(db, student, images_bytes)
    if saved_count == 0:
        raise HTTPException(status_code=400, detail="Tidak ada wajah terdeteksi pada foto.")

    return {
        "success": True,
        "message": f"Data {student.nama} berhasil disimpan.",
        "samples_saved": saved_count
    }

@app.post("/recognize")
async def recognize(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        content = await file.read()
        predictions = face_service.recognize_frame(db, content)
        return {
            "success": True,
            "total_detected": len(predictions),
            "predictions": [p.model_dump() for p in predictions]
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")