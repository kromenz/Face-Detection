# main_tracking_improved.py
import cv2
import numpy as np
import time
import face_recognition
from simple_facerec import SimpleFacerec

# -----------------------------
# Config
# -----------------------------
CAM_IDX = 0
CAP_WIDTH = 640
CAP_HEIGHT = 480

PROCESS_EVERY_N_FRAMES = 8
RECOG_INTERVAL = 3.0               
RECOG_ON_MOVEMENT_PX = 50.0
RECOG_ON_SIZE_CHANGE = 0.25 

DETECTION_SCALE = 0.25
DETECTION_MIN_SIZE = (40, 40)
DETECTION_SCALE_FACTOR = 1.1
DETECTION_MIN_NEIGHBORS = 5

DIST_LOW = 0.40               # <= low => confidence 1.0
DIST_HIGH = 0.60              # >= high => confidence 0.0
MATCH_TOLERANCE = 0.6         # fallback tolerance to mark as known (unused if using confidence)

TRACKER_TYPE = "KCF"          # options: "KCF", "CSRT" (CSRT more accurate but slower)
MAX_TRACKER_LOST_TIME = 1.2   # seconds until we drop a tracker that stopped updating
MAX_TRACKERS = 12            # limit total trackers to avoid overload

# -----------------------------
# Helpers
# -----------------------------

def create_tracker_by_name(name=TRACKER_TYPE):
    if name == "CSRT":
        return cv2.legacy.TrackerCSRT_create()
    else:
        return cv2.legacy.TrackerKCF_create()

def dist_to_confidence(d, low=DIST_LOW, high=DIST_HIGH):
    if d is None:
        return 0.0
    if d <= low:
        return 1.0
    if d >= high:
        return 0.0
    return float(np.clip((high - d) / (high - low), 0.0, 1.0))

def name_from_encoding(sfr, enc):
    if hasattr(sfr, "match_encoding"):
        return sfr.match_encoding(enc, low_threshold=DIST_LOW, high_threshold=DIST_HIGH)

    if not sfr.known_face_encodings:
        return "Unknown", None, 0.0
    dists = face_recognition.face_distance(sfr.known_face_encodings, enc)

    best_idx = np.argmin(dists)
    best_dist = float(dists[best_idx])
    best_name = sfr.known_face_names[best_idx]
    conf = dist_to_confidence(best_dist)
    if best_dist > MATCH_TOLERANCE:
        best_name = "Unknown"
    return best_name, best_dist, conf

def center_of_box(loc):
    x0, y0, x1, y1 = loc
    return ((x0 + x1) // 2, (y0 + y1) // 2)

def center_dist(loc1, loc2):
    (cx1, cy1) = center_of_box(loc1)
    (cx2, cy2) = center_of_box(loc2)
    return np.hypot(cx1 - cx2, cy1 - cy2)

def size_change_frac(loc1, loc2):
    w1 = loc1[2] - loc1[0]; h1 = loc1[3] - loc1[1]
    w2 = loc2[2] - loc2[0]; h2 = loc2[3] - loc2[1]
    if w1 == 0 or h1 == 0:
        return 1.0
    return max(abs(w2 - w1) / w1, abs(h2 - h1) / h1)

def refine_bbox_with_landmarks(frame, bbox, pad_x=0.05, pad_y_top=0.10, pad_y_bottom=0.10):
    x0, y0, x1, y1 = bbox
    h_img, w_img = frame.shape[:2]

    x0c = max(0, x0); y0c = max(0, y0)
    x1c = min(w_img, x1); y1c = min(h_img, y1)
    if x1c - x0c <= 10 or y1c - y0c <= 10:
        return (x0c, y0c, x1c, y1c)

    crop = frame[y0c:y1c, x0c:x1c]
    try:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    except Exception:
        return (x0c, y0c, x1c, y1c)

    landmarks = face_recognition.face_landmarks(rgb)
    if not landmarks:

        w = x1c - x0c
        h = y1c - y0c
        px = int(w * pad_x)
        py_top = int(h * pad_y_top)
        py_bottom = int(h * pad_y_bottom)
        nx0 = max(0, x0c - px)
        ny0 = max(0, y0c - py_top)
        nx1 = min(w_img, x1c + px)
        ny1 = min(h_img, y1c + py_bottom)
        return (nx0, ny0, nx1, ny1)

    all_x = []
    all_y = []
    for lm in landmarks:
        for part in lm.values():
            for (px, py) in part:
                all_x.append(px)
                all_y.append(py)
    if not all_x or not all_y:
        # fallback padding
        w = x1c - x0c; h = y1c - y0c
        px = int(w * pad_x)
        py_top = int(h * pad_y_top)
        py_bottom = int(h * pad_y_bottom)
        nx0 = max(0, x0c - px)
        ny0 = max(0, y0c - py_top)
        nx1 = min(w_img, x1c + px)
        ny1 = min(h_img, y1c + py_bottom)
        return (nx0, ny0, nx1, ny1)

    min_x = min(all_x); max_x = max(all_x)
    min_y = min(all_y); max_y = max(all_y)

    lm_x0 = x0c + min_x
    lm_y0 = y0c + min_y
    lm_x1 = x0c + max_x
    lm_y1 = y0c + max_y

    w_l = lm_x1 - lm_x0
    h_l = lm_y1 - lm_y0
    px = int(w_l * pad_x)
    py_top = int(h_l * pad_y_top)
    py_bottom = int(h_l * pad_y_bottom)

    nx0 = max(0, lm_x0 - px)
    ny0 = max(0, lm_y0 - py_top)
    nx1 = min(w_img, lm_x1 + px)
    ny1 = min(h_img, lm_y1 + py_bottom)

    if nx1 - nx0 < 20 or ny1 - ny0 < 20:
        nx0 = max(0, x0c - int((x1c-x0c) * pad_x))
        nx1 = min(w_img, x1c + int((x1c-x0c) * pad_x))
        ny0 = max(0, y0c - int((y1c-y0c) * pad_y_top))
        ny1 = min(h_img, y1c + int((y1c-y0c) * pad_y_bottom))

    return (int(nx0), int(ny0), int(nx1), int(ny1))

sfr = SimpleFacerec()
sfr.load_encoding_images("faces/", cache_file="cache/enc_cache.pkl", use_hash=True)   # assumes images/<person>/* structure
print("Loaded identities:", sfr.get_all_names())

cap = cv2.VideoCapture(CAM_IDX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Camera not found")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

win_name = "Tracking + Recognition"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 900, 700)

haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

face_trackers = []
next_tracker_id = 0

prev_time = time.time()
frame_idx = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame_disp = frame.copy()
        frame_idx += 1
        now = time.time()

        alive = []
        for t in face_trackers:
            success, bbox = t['tracker'].update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                loc = (x, y, x + w, y + h)
                t['loc'] = loc
                t['last_seen'] = now


                time_since_recog = now - t.get('last_recognized', 0.0)
                moved = center_dist(loc, t.get('last_loc_for_recog', loc))
                size_changed = size_change_frac(t.get('last_loc_for_recog', loc), loc)
                do_recog = False
                if t.get('name', "Unknown") == "Unknown" and time_since_recog > 0.5:
                    do_recog = True
                if time_since_recog >= RECOG_INTERVAL:
                    do_recog = True
                if moved > RECOG_ON_MOVEMENT_PX:
                    do_recog = True
                if size_changed > RECOG_ON_SIZE_CHANGE:
                    do_recog = True

                if do_recog:
                    h_img, w_img = frame.shape[:2]
                    x0, y0, x1, y1 = max(0, loc[0]), max(0, loc[1]), min(w_img, loc[2]), min(h_img, loc[3])
                    if x1 - x0 > 10 and y1 - y0 > 10:
                        face_crop = frame[y0:y1, x0:x1]
                        try:
                            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                            rgb_face = np.ascontiguousarray(rgb_face)
                            # compute encodings directly on the ROI
                            encs = face_recognition.face_encodings(rgb_face)
                        except Exception:
                            encs = []
                        if encs:
                            enc = encs[0]
                            name, dist, conf = name_from_encoding(sfr, enc)
                            t['name'] = name
                            t['dist'] = dist
                            t['conf'] = conf
                        t['last_recognized'] = now
                        t['last_loc_for_recog'] = loc

                label = f"{t.get('name','Unknown')} {int(t.get('conf',0)*100)}%"
                x0, y0, x1, y1 = t['loc']
                cv2.rectangle(frame_disp, (x0, y0), (x1, y1), (0, 200, 0), 2)
                cv2.putText(frame_disp, label, (x0, max(10,y0 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

                alive.append(t)
            else:
                if now - t.get('last_seen', 0) < MAX_TRACKER_LOST_TIME:
                    alive.append(t)

        face_trackers = alive

        if frame_idx % PROCESS_EVERY_N_FRAMES == 0 and len(face_trackers) < MAX_TRACKERS:
            small = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            dets = haar.detectMultiScale(gray, scaleFactor=DETECTION_SCALE_FACTOR,
                                         minNeighbors=DETECTION_MIN_NEIGHBORS, minSize=DETECTION_MIN_SIZE)
            for (x, y, w, h) in dets:

                x0, y0, x1, y1 = int(x / DETECTION_SCALE), int(y / DETECTION_SCALE), int((x + w) / DETECTION_SCALE), int((y + h) / DETECTION_SCALE)
                new_loc = (x0, y0, x1, y1)

                matched = False
                for t in face_trackers:
                    if center_dist(new_loc, t['loc']) < max(40, (x1-x0)/2):
                        matched = True
                        break
                if matched:
                    continue

                h_img, w_img = frame.shape[:2]
                x0c, y0c = max(0, x0), max(0, y0)
                x1c, y1c = min(w_img, x1), min(h_img, y1)
                if x1c - x0c <= 10 or y1c - y0c <= 10:
                    continue

                ref_x0, ref_y0, ref_x1, ref_y1 = refine_bbox_with_landmarks(frame, (x0c,y0c,x1c,y1c),
                                                            pad_x=0.18, pad_y_top=0.18, pad_y_bottom=0.28)

                face_crop = frame[ref_y0:ref_y1, ref_x0:ref_x1]

                name, dist, conf = "Unknown", None, 0.0
                
                try:
                    if face_crop.size > 0:
                        rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        rgb_face = np.ascontiguousarray(rgb_face)
                        face_locations, encs = sfr.detect_faces(face_crop)

                        if not encs:
                            small_face = cv2.resize(face_crop, (0, 0), fx=0.5, fy=0.5)
                            rgb_small = cv2.cvtColor(small_face, cv2.COLOR_BGR2RGB)
                            locs_small = face_recognition.face_locations(rgb_small, model="hog")
                            encs = face_recognition.face_encodings(rgb_small, locs_small)

                        if encs:
                            enc = encs[0]
                            name, dist, conf = name_from_encoding(sfr, enc)

                except Exception as e:
                    encs = []
                    
                try:
                    tracker = create_tracker_by_name(TRACKER_TYPE)
                    tracker.init(frame, (ref_x0, ref_y0, ref_x1 - ref_x0, ref_y1 - ref_y0))
                except Exception:
                    continue

                tdict = {
                    'id': next_tracker_id,
                    'tracker': tracker,
                    'loc': (x0c, y0c, x1c, y1c),
                    'name': name,
                    'dist': dist,
                    'conf': conf,
                    'last_seen': now
                }
                next_tracker_id += 1
                face_trackers.append(tdict)

        tnow = time.time()
        fps = 1.0 / (tnow - prev_time) if (tnow - prev_time) > 0 else 0.0
        prev_time = tnow
        cv2.putText(frame_disp, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow(win_name, frame_disp)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
