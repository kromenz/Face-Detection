import cv2

alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + alg)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

win_name = "Face Detected"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 1024, 768)

scaleFactor = 1.1
minNeighbors = 8
minSize = (100, 100)     
upscale = 1.5          

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not found")
        continue

    h, w = frame.shape[:2]

    det_frame = cv2.resize(frame, (int(w*upscale), int(h*upscale)),
                           interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = haar_cascade.detectMultiScale(gray,
                                          scaleFactor=scaleFactor,
                                          minNeighbors=minNeighbors,
                                          minSize=minSize)

    for (x, y, fw, fh) in faces:
        x0 = int(x / upscale); y0 = int(y / upscale)
        x1 = int((x + fw) / upscale); y1 = int((y + fh) / upscale)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

    cv2.imshow(win_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
