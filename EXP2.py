import cv2
from keras.models import model_from_json
import numpy as np
# from keras_preprocessing.image import load_img
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("emotiondetector.h5")
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0
# # Open a video fileimport cv2
from keras.models import model_from_json
import numpy as np

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("emotiondetector.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

video_path = '3308.mp4'
webcam = cv2.VideoCapture(video_path)

if not webcam.isOpened():
    print("Error: Could not open video file.")
    exit()

labels = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}

while True:
    ret, im = webcam.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize the frame to make it smaller
    im = cv2.resize(im, (640, 480))

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)

    try:
        for (p, q, r, s) in faces:
            image = gray[q:q + s, p:p + r]
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.putText(im, f'Emotion: {prediction_label}', (p, q + s + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                        2)

        cv2.imshow("Output", im)

        # Adjust the waitKey parameter to control the display speed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except cv2.error:
        pass

webcam.release()
cv2.destroyAllWindows()

