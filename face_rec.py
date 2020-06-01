import face_recognition
import os
import cv2
from google.colab.patches import cv2_imshow

KNOWN_FACES  = "/content/facerec/known_faces"
UNKNOWN_FACES = "/content/facerec/unknown_faces"
TOLERENCE  = 0.5
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "cnn"
print("loading known faces")
known_faces = []
known_names =[]

for name in os.listdir(KNOWN_FACES):
  for filename in os.listdir(f"{KNOWN_FACES}/{name}"):
    image = face_recognition.load_image_file(f"{KNOWN_FACES}/{name}/{filename}")
    encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(encoding)
    known_names.append(name)

print("processing unkmonw faces")
for filename in os.listdir(UNKNOWN_FACES):
  print(filename)
  image = face_recognition.load_image_file(f"{UNKNOWN_FACES}/{filename}")
  locations = face_recognition.face_locations(image, model=MODEL)
  encodings = face_recognition.face_encodings(image, locations)

  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  
  for face_encoding, face_location in zip(encodings, locations):
    results = face_recognition.compare_faces(known_faces, face_encoding, TOLERENCE)
    match = None
    if True in results:
      match = known_names[results.index(True)]
      print(f"Match found: {match}")

      top_left = (face_location[3], face_location[0])
      bottom_right = (face_location[1], face_location[2])

      color = [0, 255, 0]

      cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

      top_left = (face_location[3], face_location[2])
      top_left = (face_location[1], face_location[2]+22)
      cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
      cv2.putText(image, match, (face_location[3]+10 , face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), FONT_THICKNESS)
  cv2_imshow(image)
  cv2.waitKey(1)
 # cv2.destroyWindow(filename)
