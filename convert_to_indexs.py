import os
import json
import pickle
import hashlib
import numpy as np
import face_recognition
import faiss
import time 

start = time.time()
# === Config ===
BASE_DIR = "org_images/Emp_images"
INDEX_FILE = "user_face_index.index"
META_FILE = "user_face_metadata.pkl"
CACHE_FILE = "index_cache.json"

# === Helpers ===
def hash_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# === Load cache ===
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        image_cache = json.load(f)
else:
    image_cache = {}

# === Load existing FAISS index and metadata ===
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)
else:
    index = None
    metadata = []

# === Collect new encodings and metadata ===
new_encodings = []
new_metadata = []

for user_id in os.listdir(BASE_DIR):
    user_path = os.path.join(BASE_DIR, user_id)
    if not os.path.isdir(user_path):
        continue

    # Load user's metadata
    json_path = os.path.join(user_path, "data.json")
    user_data = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            user_data = json.load(f)

    for filename in os.listdir(user_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        file_path = os.path.join(user_path, filename)
        file_hash = hash_file(file_path)

        # Skip if already processed
        if image_cache.get(file_path) == file_hash:
            continue

        print(f"🔍 Processing: {file_path}")
        image = face_recognition.load_image_file(file_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            new_encodings.append(encodings[0])
            new_metadata.append({
                **user_data,
                "user_id": user_id,
                "image_file": filename,
                "image_path": file_path
            })
            image_cache[file_path] = file_hash
            print(f"✅ Indexed: {user_id}/{filename}")
            # Immediately update cache file
            with open(CACHE_FILE, "w") as f:
                json.dump(image_cache, f)
        else:
            print(f"⚠️ No face found in: {file_path}")

# === Update index and metadata ===
if new_encodings:
    vecs = np.array(new_encodings).astype("float32")

    if index is None:
        index = faiss.IndexFlatL2(vecs.shape[1])

    index.add(vecs)
    metadata.extend(new_metadata)

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)

    print(f"\n🎉 Indexed {len(new_encodings)} new face(s).")
else:
    print("⏩ No new faces to index.")

# # === Save updated cache ===
# with open(CACHE_FILE, "w") as f:
#     json.dump(image_cache, f)

end = time.time()

print(f"⏱️ Total time taken: {end - start:.2f} seconds")
