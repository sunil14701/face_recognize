import os
import json
import pickle
import numpy as np
import face_recognition
import faiss

BASE_DIR = "org_images/Emp_images"
INDEX_FILE = "user_face_index.index"
META_FILE = "user_face_metadata.pkl"

all_encodings = []
metadata = []

for user_id in os.listdir(BASE_DIR):
    user_path = os.path.join(BASE_DIR, user_id)
    if not os.path.isdir(user_path):
        continue

    # Load user's metadata
    json_path = os.path.join(user_path, "data.json")
    user_metadata = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            user_metadata = json.load(f)

    for filename in os.listdir(user_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        file_path = os.path.join(user_path, filename)
        print(f"🔍 Processing: {file_path}")

        try:
            image = face_recognition.load_image_file(file_path)
            encodings = face_recognition.face_encodings(image)
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            continue

        if encodings:
            all_encodings.append(encodings[0])
            metadata.append({
                **user_metadata,
                "user_id": user_id,
                "image_file": filename,
                "image_path": file_path
            })
            print(f"✅ Indexed: {user_id}/{filename}")
        else:
            print(f"⚠️ No face found in: {file_path}")

# Create FAISS index
if all_encodings:
    enc_array = np.array(all_encodings).astype("float32")
    index = faiss.IndexFlatL2(enc_array.shape[1])
    index.add(enc_array)

    # Save index
    faiss.write_index(index, INDEX_FILE)

    # Save metadata
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)

    print(f"\n🎉 Done! Indexed {len(all_encodings)} face(s).")
    print(f"📦 FAISS index saved to: {INDEX_FILE}")
    print(f"🗂️  Metadata saved to: {META_FILE}")
else:
    print("❗ No faces found. Nothing was indexed.")
