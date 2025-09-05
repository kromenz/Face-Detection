import cv2
import numpy as np
from simple_facerec import SimpleFacerec
import time

sfr = SimpleFacerec()
sfr.load_encoding_images("faces/")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

win_name = "Face Detection with Tracking"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 800, 600)

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
scaleFactor = 1.1
minNeighbors = 5
minSize = (40, 40)

face_trackers = []

prev_time = time.time()
frame_count = 0
process_every_n_frames = 10 

def create_tracker():
    return cv2.legacy.TrackerKCF_create() 

def euclidean_dist(loc1, loc2):
    x0, y0, x1, y1 = loc1
    x0b, y0b, x1b, y1b = loc2
    cx1, cy1 = (x0 + x1)//2, (y0 + y1)//2
    cx2, cy2 = (x0b + x1b)//2, (y0b + y1b)//2
    return ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame_disp = frame.copy()
    frame_count += 1

    new_trackers = []
    for f in face_trackers:
        success, bbox = f['tracker'].update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            f['loc'] = (x, y, x + w, y + h)
            new_trackers.append(f)
            cv2.rectangle(frame_disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_disp, f['name'], (x, y - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
    face_trackers = new_trackers

    if frame_count % process_every_n_frames == 0:
        small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray, scaleFactor=scaleFactor,
                                              minNeighbors=minNeighbors,
                                              minSize=minSize)
        for (x, y, w, h) in faces:
            x0, y0, x1, y1 = int(x*4), int(y*4), int((x+w)*4), int((y+h)*4)
            new_face = True
            for f in face_trackers:
                if euclidean_dist((x0,y0,x1,y1), f['loc']) < 50:
                    new_face = False
                    break
            if new_face:
                face_crop = frame[y0:y1, x0:x1]
                name = "Unkwnown"
                if face_crop.size != 0:
                    rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    rgb_face = np.ascontiguousarray(rgb_face)
                    encodings = sfr.detect_faces(face_crop)[1]
                    if encodings:
                        name = sfr.compare_faces(encodings[0])
                tracker = create_tracker()
                tracker.init(frame, (x0, y0, x1-x0, y1-y0))
                face_trackers.append({'tracker': tracker, 'name': name, 'loc': (x0, y0, x1, y1)})

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame_disp, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow(win_name, frame_disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
