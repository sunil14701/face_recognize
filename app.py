from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import face_recognition
import faiss
import pickle
import os
import io
from PIL import Image, ExifTags

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or use your frontend origin
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Config ===
INDEX_FILE = "user_face_index.index"
META_FILE = "user_face_metadata.pkl"
TOP_K = 3

# === Load FAISS + Metadata once on startup ===
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    metadata = pickle.load(f)

def prepare_image(img_bytes, max_width=1000):
    img = Image.open(io.BytesIO(img_bytes))

    # Apply EXIF rotation
    try:
        for orientation in ExifTags.TAGS:
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation, None)
            if orientation_value == 3:
                img = img.rotate(180, expand=True)
            elif orientation_value == 6:
                img = img.rotate(270, expand=True)
            elif orientation_value == 8:
                img = img.rotate(90, expand=True)
    except Exception:
        pass  # Skip if no EXIF

    # Convert to RGB
    img = img.convert("RGB")

    # Resize if too large
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)))

    return img

@app.post("/search/")
async def identify_person(file: UploadFile = File(...)):
    print("📦 Received file:", file.filename, file.content_type)
    # Load image into memory
    img_bytes = await file.read()
    img = prepare_image(img_bytes)
    
    # Convert back to numpy array
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_np = face_recognition.load_image_file(io.BytesIO(buf.getvalue()))

    face_locations = face_recognition.face_locations(image_np)
    if not face_locations:
        return JSONResponse(status_code=400, content={"error": "No face found in image."})


    # Encode face
    encodings = face_recognition.face_encodings(image_np)
    print(encodings)
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

#  uvicorn app:app --port 8000 --host 0.0.0.0