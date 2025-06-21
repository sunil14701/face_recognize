from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import face_recognition
import faiss
import pickle
import os
import io

app = FastAPI()

# === Config ===
INDEX_FILE = "user_face_index.index"
META_FILE = "user_face_metadata.pkl"
TOP_K = 3

# === Load FAISS + Metadata once on startup ===
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    metadata = pickle.load(f)

@app.post("/search/")
async def identify_person(file: UploadFile = File(...)):
    # Load image into memory
    img_bytes = await file.read()
    image = face_recognition.load_image_file(io.BytesIO(img_bytes))

    # Encode face
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        return JSONResponse(status_code=400, content={"error": "No face found in the uploaded image."})

    vec = np.array([encodings[0]]).astype("float32")
    D, I = index.search(vec, TOP_K)

    for dist, idx in zip(D[0], I[0]):
        if dist > 1e6 or idx >= len(metadata):
            continue
        meta = metadata[idx]

        # Add distance to metadata before returning
        result = dict(meta)
        result["distance"] = float(dist)
        return result

    return JSONResponse(status_code=404, content={"error": "No confident match found."})

#  uvicorn main:app