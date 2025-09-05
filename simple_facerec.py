import face_recognition
import cv2
import os
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def load_encoding_images(self, folder_path):
        images = os.listdir(folder_path)
        print(f"Found {len(images)} images in '{folder_path}' folder.")

        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️  Image {img_name} couldn't be read. Skipping...")
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_img = np.ascontiguousarray(rgb_img)

            face_locations = face_recognition.face_locations(rgb_img)
            if not face_locations:
                print(f"⚠️  No face found in {img_name}, skipping...")
                continue

            encodings = face_recognition.face_encodings(rgb_img, known_face_locations=face_locations)
            self.known_face_encodings.append(encodings[0])
            name = os.path.splitext(img_name)[0]
            self.known_face_names.append(name)
            print(f"✅  Encoded face: {name}")

        print(f"\nTotal available faces: {len(self.known_face_encodings)}")

    def compare_faces(self, face_encoding):
        if not self.known_face_encodings:
            return "Unknown"

        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

        name = "Unknown"
        if matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
        return name
    
    def detect_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        return face_locations, face_encodings

