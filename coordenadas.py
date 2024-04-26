"codigo para mostrar como se crean las coordenadas"

import cv2
import face_recognition

# Cargar imagen de referencia
image_path = "/home/valenzzabala/Descargas/944261ba-3c74-4a4d-88d7-e18006589301.jpeg"
reference_image = cv2.imread(image_path)

# Detectar rostros en la imagen de referencia
face_locations_reference = face_recognition.face_locations(reference_image)
print("Coordenadas de los rostros en la imagen de referencia:")
for face_location in face_locations_reference:
    print(face_location)

# Codificar el primer rostro detectado en la imagen de referencia
reference_face_encoding = face_recognition.face_encodings(reference_image, known_face_locations=[face_locations_reference[0]])[0]

# Mostrar la codificación del rostro
print("Codificación del rostro en la imagen de referencia:")
print(reference_face_encoding)
