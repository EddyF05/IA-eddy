# detect_faces.py
import cv2
import os

cascade_path = os.path.join("haarcascades", "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)  # 0 = webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
