from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from playsound import playsound 
import numpy as np
import cv2
import os
import time 

print("[INFO] loading models...")
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
model = load_model("face_mask_detector.h5")

ALERT_COOLDOWN = 3
last_alert_time = 0

print("[INFO] starting video stream...")
video_capture = cv2.VideoCapture(0)

while True:
	ret, frame = video_capture.read()
	if not ret:
		break

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

	for (x, y, w, h) in faces:
		face_roi = frame[y:y+h, x:x+w]
		face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
		face_roi = cv2.resize(face_roi, (224, 224))
		face_roi = img_to_array(face_roi)
		face_roi = preprocess_input(face_roi)
		face_roi = np.expand_dims(face_roi, axis=0)

		(mask, withoutMask) = model.predict(face_roi)[0]

		label_text = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label_text == "Mask" else (0, 0, 255)
		if label_text == "No Mask":
			current_time = time.time()
			if (current_time - last_alert_time) > ALERT_COOLDOWN:
				last_alert_time = current_time
				playsound('alert.wav', block=False)

		label_display = "{}: {:.2f}%".format(label_text, max(mask, withoutMask) * 100)
		cv2.putText(frame, label_display, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

	cv2.imshow("Face Mask Detector", frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

print("[INFO] cleaning up...")
video_capture.release()
cv2.destroyAllWindows()