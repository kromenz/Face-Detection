import cv2
import os

alg = "haarcascade_frontalface_default.xml"

haar_cascade=cv2.CascadeClassifier(alg)
file_name="examples/IMG_1051.jpeg"
img = cv2.imread(file_name)
if img is None:
    raise FileNotFoundError(f"Image not found: {file_name}")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

faces=haar_cascade.detectMultiScale(gray_img,scaleFactor=1.05,minNeighbors=5, minSize=(100,100))

os.makedirs("output", exist_ok=True)
for i, (x, y, w, h) in enumerate(faces):
    cropped_img=img[y:y+h,x:x+w]
    target_file_name="output/face"+str(i)+".jpg"
    cv2.imwrite(target_file_name,cropped_img)
    i = i +1
