#Source https://pysource.com/2018/10/01/face-detection-using-haar-cascades-opencv-3-4-with-python-3-tutorial-37/

#Importa as bibliotecas do openCV e do Numpy
import cv2
import numpy as np

#Captura a imagem da webcam
cap = cv2.VideoCapture(0)
#Importa os modelos xml dos cascades
frontal_face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
profile_face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_profileface.xml")

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

while True:
    #Le um frame da camera
    ret , frame = cap.read()
    #Transforma em escala de cinza (melhor para o reconheciento vai haar cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Detecta as faces de frente e de perfil
    front_faces = frontal_face_cascade.detectMultiScale(gray, 1.3, 3)
    profile_faces = profile_face_cascade.detectMultiScale(gray, 1.3, 3)

    #Desenha quadrados verdes dobre todas as faces frontais
    for rect in front_faces:
        (x, y, w, h) = rect
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #Desenha quadrados azuis sobre todas as faces de perfil
    for rect in profile_faces:
         (x, y, w, h) = rect
         frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

    #Mostra o frame com eventuais rostos detectados
    cv2.imshow("Frame", frame)
    #Se pressionar Esc ( = key 27), fecha a janela
    key = cv2.waitKey(1)
    if key == 27:
        break

#Libera a camera e fecha as janelas
cap.release()
cv2.destroyAllWindows()