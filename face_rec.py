"""Implementation of face recognition with python face recognition libraray"""
#importing necessery ibraries
import face_recognition
import os
import cv2
from google.colab.patches import cv2_imshow # for showing images in google colab

KNOWN_FACES  = "/content/facerec/known_faces" # directory for known facs
UNKNOWN_FACES = "/content/facerec/unknown_faces" #directory for unknown faces
TOLERENCE  = 0.5 #sensitivity for comparing faces
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "cnn" # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model
print("loading known faces")
known_faces = []
known_names =[]

for name in os.listdir(KNOWN_FACES): #iterating over each directory in known_faces directoy=ry
  for filename in os.listdir(f"{KNOWN_FACES}/{name}"): #iteraating over each image in name directory
    image = face_recognition.load_image_file(f"{KNOWN_FACES}/{name}/{filename}")
    encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(encoding) # appending the encodings of images to an array
    known_names.append(name) # appending the corresponding name for the encodings

print("processing unkmonw faces") # processing for unknown images
for filename in os.listdir(UNKNOWN_FACES): #iterating over each image in unknown_faces directory
  print(filename)
  image = face_recognition.load_image_file(f"{UNKNOWN_FACES}/{filename}")
  locations = face_recognition.face_locations(image, model=MODEL)
  encodings = face_recognition.face_encodings(image, locations)

  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  
  for face_encoding, face_location in zip(encodings, locations):
    results = face_recognition.compare_faces(known_faces, face_encoding, TOLERENCE)
    match = None  # initilizing the match as none
    if True in results: # if any match found in results then
      match = known_names[results.index(True)]
      print(f"Match found: {match}")
      
"""creating boxes around the face"""
      top_left = (face_location[3], face_location[0]) 
      bottom_right = (face_location[1], face_location[2])

      color = [0, 255, 0]

      cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

      top_left = (face_location[3], face_location[2])
      top_left = (face_location[1], face_location[2]+22)
      cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
      cv2.putText(image, match, (face_location[3]+10 , face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), FONT_THICKNESS)
  cv2_imshow(image) #showing the image in google colab else use cv2.imshow(filename, image)
  cv2.waitKey(1) # waitkey of 1ms and iterating over images
 # cv2.destroyWindow(filename)
