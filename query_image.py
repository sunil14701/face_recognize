import face_recognition
import faiss
import numpy as np
import pickle

# === CONFIG ===
INPUT_IMAGE_PATH = "org_images/search_person/test.jpg"
INDEX_PATH = "face_index.index"
NAME_MAP_PATH = "name_map.pkl"
TOP_K = 3

# === Load Index and Names ===
try:
    index = faiss.read_index(INDEX_PATH)
except Exception as e:
    print(f"❌ Failed to load FAISS index: {e}")
    exit()

try:
    with open(NAME_MAP_PATH, "rb") as f:
        name_list = pickle.load(f)
except Exception as e:
    print(f"❌ Failed to load name mapping: {e}")
    exit()

# === Load and Encode Input Image ===
unknown_image = face_recognition.load_image_file(INPUT_IMAGE_PATH)
encodings = face_recognition.face_encodings(unknown_image)

if not encodings:
    print("❌ No face found in the input image.")
    exit()

unknown_vector = np.array([encodings[0]]).astype("float32")

if unknown_vector.shape != (1, 128):
    print("❌ Invalid encoding shape. Expected (1, 128), got", unknown_vector.shape)
    exit()

# === Search in FAISS ===
D, I = index.search(unknown_vector, TOP_K)

# === Display Results ===
print("\n🎯 Top Matches:")
for dist, idx in zip(D[0], I[0]):
    if dist > 1e6 or idx >= len(name_list):
        continue  # skip invalid results
    name = name_list[idx]
    print(f"✅ Match: {name}, Distance: {dist:.4f}")
