import cv2
import numpy as np
from deepface import DeepFace
import pyttsx3
import os
import time
engine = pyttsx3.init()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
db_path = "C:/Users/Dusky Doll/Desktop/Open CV Projects/Varshitha/"
cap = cv2.VideoCapture(0)
recognized_last = None
last_announcement_time = 0
announcement_interval = 3 
SIMILARITY_THRESHOLD = 0.4  
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))
    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]  
        recognized_name = "Unknown arrived"  
        try:
            result = DeepFace.find(face_crop, db_path=db_path, enforce_detection=False)

            if len(result) > 0 and len(result[0]) > 0:
                matched_file = result[0]['identity'][0]  
                similarity_score = result[0]['distance'][0] 
                matched_person = os.path.basename(os.path.dirname(matched_file))
                if similarity_score < SIMILARITY_THRESHOLD:
                    recognized_name = matched_person 
                else:
                    recognized_name = "Unknown arrived"  
        except Exception as e:
            print(f"Error in DeepFace processing: {e}")
        current_time = time.time()
        if recognized_name != recognized_last or (current_time - last_announcement_time > announcement_interval):
            print(f"Recognized: {recognized_name}")
            engine.say(f"{recognized_name} arrived")
            engine.runAndWait()
            recognized_last = recognized_name
            last_announcement_time = current_time

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
