import face_recognition
import numpy as np
import faiss
import os
import pickle

known_dir = "org_images/train_data"
name_list = []
encoding_list = []

for file in os.listdir(known_dir):
    path = os.path.join(known_dir, file)
    img = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(img)
    if encodings:
        encoding_list.append(encodings[0])
        name_list.append(os.path.splitext(file)[0])

# Save FAISS index
vecs = np.array(encoding_list).astype('float32')
index = faiss.IndexFlatL2(128)
index.add(vecs)

faiss.write_index(index, "face_index.index")

# Save names
with open("name_map.pkl", "wb") as f:
    pickle.dump(name_list, f)
