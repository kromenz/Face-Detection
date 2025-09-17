import csv
import hashlib
import pickle
import tempfile
import face_recognition
import cv2
import os
import numpy as np
from datetime import datetime

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self._cache = {} 

    def _file_key(self, path, use_hash=True):
        if use_hash:
            h = hashlib.md5()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        else:
            return os.path.getmtime(path)
    
    def load_cache(self, cache_file):
        if not cache_file or not os.path.isfile(cache_file):
            return
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                self._cache = data
                print(f"Loaded cache with {len(self._cache)} items from '{cache_file}'")
        except Exception as e:
            print(f"⚠️  Couldn't load cache '{cache_file}': {e}")

    def save_cache(self, cache_file):
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        if not cache_file:
            return
        try:
            dirn = os.path.dirname(os.path.abspath(cache_file)) or "."
            with tempfile.NamedTemporaryFile(dir=dirn, delete=False) as tf:
                pickle.dump(self._cache, tf)
                tempname = tf.name
            os.replace(tempname, cache_file)
            print(f"Saved cache ({len(self._cache)} items) to '{cache_file}'")
        except Exception as e:
            print(f"⚠️  Couldn't save cache '{cache_file}': {e}")

    def clear_cache(self):
        self._cache = {}

    def remove_from_cache(self, img_path):
        if img_path in self._cache:
            del self._cache[img_path]
            
    def load_encoding_images(self, folder_path,
                             cache_file="enc_cache.pkl",
                             allowed_ext=(".jpg", ".jpeg", ".png"),
                             resize_max_width=1200,
                             one_encoding_per_image=True,
                             use_hash=False):

        if cache_file:
            self.load_cache(cache_file)
        
        self.known_face_encodings = []
        self.known_face_names = []

        if not os.path.isdir(folder_path):
            print(f"⚠️  Folder '{folder_path}' does not exist.")
            return

        persons = [d for d in os.listdir(folder_path)
                   if os.path.isdir(os.path.join(folder_path, d))]
        if not persons:
            print(f"⚠️  No subfolders found in '{folder_path}'.")
            return

        total_images = 0
        failed = []

        for person in sorted(persons):
            person_label = person.replace("_", " ").strip()
            person_path = os.path.join(folder_path, person)
            images = [f for f in os.listdir(person_path)
                      if os.path.isfile(os.path.join(person_path, f)) and
                      os.path.splitext(f)[1].lower() in allowed_ext]

            if not images:
                print(f"⚠️  No valid images in '{person_path}'. Skipping.")
                continue

            ok_count = 0
            for img_name in images:
                img_path = os.path.abspath(os.path.join(person_path, img_name))
                try:
                    key = self._file_key(img_path, use_hash=use_hash)
                except Exception as e:
                    failed.append((person_label, img_name, f"key-error: {e}"))
                    continue

                cache_entry = self._cache.get(img_path)
                if cache_entry and cache_entry.get('key') == key:
                    # cache hit: reuse encoding
                    enc = cache_entry.get('encoding')
                    name = cache_entry.get('name', person_label)

                    if isinstance(enc, list):
                        for e in enc:
                            self.known_face_encodings.append(e)
                            self.known_face_names.append(name)
                            ok_count += 1
                            total_images += 1
                    else:
                        self.known_face_encodings.append(enc)
                        self.known_face_names.append(name)
                        ok_count += 1
                        total_images += 1

                    continue
                
                # cache miss: compute encoding
                img = cv2.imread(img_path)
                if img is None:
                    failed.append((person_label, img_name, "imread returned None"))
                    continue

                if resize_max_width and img.shape[1] > resize_max_width:
                    scale = resize_max_width / img.shape[1]
                    new_h = int(img.shape[0] * scale)
                    img = cv2.resize(img, (resize_max_width, new_h), interpolation=cv2.INTER_AREA)

                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rgb_img = np.ascontiguousarray(rgb_img)

                try:
                    face_locations = face_recognition.face_locations(rgb_img, model="hog")
                except Exception as e:
                    failed.append((person_label, img_name, f"face_locations error: {e}"))
                    continue

                if not face_locations:
                    failed.append((person_label, img_name, "no face detected"))
                    continue

                try:
                    encodings = face_recognition.face_encodings(rgb_img, known_face_locations=face_locations)
                except Exception as e:
                    failed.append((person_label, img_name, f"encodings error: {e}"))
                    continue

                if not encodings:
                    failed.append((person_label, img_name, "no encodings"))
                    continue

                if one_encoding_per_image:
                    enc_to_store = encodings[0]
                    self.known_face_encodings.append(enc_to_store)
                    self.known_face_names.append(person_label)
                    # update cache
                    self._cache[img_path] = {'key': key, 'encoding': enc_to_store, 'name': person_label}
                    ok_count += 1
                    total_images += 1
                else:
                    for enc in encodings:
                        self.known_face_encodings.append(enc)
                        self.known_face_names.append(person_label)
                    # store all encs as list
                    self._cache[img_path] = {'key': key, 'encoding': encodings, 'name': person_label}
                    ok_count += len(encodings)
                    total_images += len(encodings)

            print(f"✅  {person_label}: {ok_count}/{len(images)} images encoded.")

        # save cache (atomic)
        if cache_file:
            try:
                self.save_cache(cache_file)
            except Exception as e:
                print(f"⚠️  Error saving cache: {e}")

        if failed:
            print("\nSome images failed to encode (see list):")
            for p, im, reason in failed:
                print(f" - {p}/{im}: {reason}")

        print(f"\nTotal encoded images: {total_images}")
        print(f"Total identities available: {len(set(self.known_face_names))}")
        
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

    def get_all_names(self):
        return sorted(set(self.known_face_names))

    def _name_to_indices(self):
        idxs = {}
        for i, n in enumerate(self.known_face_names):
            idxs.setdefault(n, []).append(i)
        return idxs
    
    def match_encoding(self, face_encoding, low_threshold=0.40, high_threshold=0.60):

        if not self.known_face_encodings:
            return "Unknown", None, 0.0

        distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        name_to_indices = self._name_to_indices()

        best_name = "Unknown"
        best_dist = float("inf")

        for name, indices in name_to_indices.items():
            dists = distances[indices]
            dmin = float(np.min(dists))
            if dmin < best_dist:
                best_dist = dmin
                best_name = name

        if best_dist is None:
            confidence = 0.0
        else:
            if best_dist <= low_threshold:
                confidence = 1.0
            elif best_dist >= high_threshold:
                confidence = 0.0
            else:
                confidence = (high_threshold - best_dist) / (high_threshold - low_threshold)
                confidence = float(np.clip(confidence, 0.0, 1.0))

        return best_name, float(best_dist), confidence