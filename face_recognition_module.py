import os
import cv2
import face_recognition

def load_known_faces(path='Training_images'):
    images = []
    class_names = []
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file))
        if img is not None:
            images.append(img)
            class_names.append(os.path.splitext(file)[0])
    return images, class_names

def encode_faces(images):
    encoded_list = []
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(rgb)[0]
            encoded_list.append(encode)
        except IndexError:
            continue
    return encoded_list

def recognize_faces(frame, known_encodings, class_names):
    rgb_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(rgb_small, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    recognized = []
    for encode, loc in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, encode)
        face_dist = face_recognition.face_distance(known_encodings, encode)

        if matches:
            best_match = face_dist.argmin()
            if matches[best_match]:
                name = class_names[best_match].upper()
                recognized.append((name, loc))
    return recognized
