import cv2
import face_recognition

# Cargar imagen de referencia
image_path = "/home/valenzzabala/Descargas/944261ba-3c74-4a4d-88d7-e18006589301.jpeg"
reference_image = cv2.imread(image_path)

# Localizar el rostro en la imagen de referencia
reference_face_location = face_recognition.face_locations(reference_image)[0]
# Codificar el rostro para comparación
reference_face_encoding = face_recognition.face_encodings(reference_image, known_face_locations=[reference_face_location])[0]


# Configuración de la captura de video
cap = cv2.VideoCapture(0)  # Iniciar la captura de video desde la cámara predeterminada (0)
cap.set(cv2.CAP_PROP_FPS,10)  # Reducir la frecuencia de fotogramas a 10 FPS

while True:
    ret, frame = cap.read()  # Leer un fotograma de la cámara
    if not ret:
        break

    frame = cv2.flip(frame,1)  # Voltear el fotograma horizontalmente (efecto espejo)

    # Detección facial en el fotograma actual utilizando el modelo HOG
    face_locations = face_recognition.face_locations(frame, model="hog")
    for face_location in face_locations:
        # Codificar el rostro detectado para comparación
        face_encoding = face_recognition.face_encodings(frame, known_face_locations=[face_location])[0]
        # Comparar la codificación del rostro detectado con la codificación de la imagen de referencia
        result = face_recognition.compare_faces([reference_face_encoding], face_encoding)

        # Identificar y dibujar el cuadro del rostro
        color = (125, 220, 0) if result[0] else (50, 50, 255)
        cv2.rectangle(frame, (face_location[3], face_location[0]), (face_location[1], face_location[2]), color, 2)

        # Mostrar resultado
        name = "Valentin" if result[0] else "Desconocido"
        text_size, _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (face_location[3], face_location[2] + 10),
                      (face_location[3] + text_size[0] + 20, face_location[2] + text_size[1] + 30), color, -1)
        cv2.putText(frame, name, (face_location[3] + 10, face_location[2] + text_size[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Mostrar el fotograma con las detecciones
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Esperar por la tecla 'Esc' para salir
        break

# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()

