import face_recognition
import faiss
import numpy as np
import pickle
import json
import os

# Constants
INDEX_FILE = "user_face_index.index"
META_FILE = "user_face_metadata.pkl"
INPUT_IMAGE = "org_images/search_person/abhinav.jpg"
TOP_K = 3

# Load index + metadata
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    metadata = pickle.load(f)

# Load and encode input image
image = face_recognition.load_image_file(INPUT_IMAGE)
encodings = face_recognition.face_encodings(image)

if not encodings:
    print("❌ No face found in input image.")
    exit()

vec = np.array([encodings[0]]).astype("float32")

# Search in index
D, I = index.search(vec, TOP_K)

print("\n🎯 Top Matches:")
for dist, idx in zip(D[0], I[0]):
    if dist > 1e6 or idx >= len(metadata):
        continue

    meta = metadata[idx]
    print(f"\n✅ Match Found: User ID: {meta['user_id']}")
    print(f"📷 File: {meta['image_file']} | 📏 Distance: {dist:.4f}")
    print("\n📄 User Metadata:")
    print(json.dumps(meta, indent=4))
    break  # only show top 1 match
