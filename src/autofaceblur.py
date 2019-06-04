#Source https://pysource.com/2018/10/01/face-detection-using-haar-cascades-opencv-3-4-with-python-3-tutorial-37/

#Importa as bibliotecas do openCV e do Numpy
import time

import cv2
import numpy as np

from tracker import faceObject
from tracker import faceTracker

#Captura a imagem da webcam
cap = cv2.VideoCapture(0)
#Importa os modelos xml dos cascades
frontal_face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
#frontal_face_cascade = cv2.CascadeClassifier("cascades/myCascade2.xml")

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

tracker = faceTracker()

fps = 30.0

while True:
    #Conta o tempo de inicio do ciclo
    tempo_inicio = time.time()
    #Le um frame da camera
    ret , frame = cap.read()
    #Transforma em escala de cinza (melhor para o reconhecimento vai haar cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Detecta as faces de frente e de perfil
    front_faces = frontal_face_cascade.detectMultiScale(gray, 1.3, 3)

    frameFacesList = list()

    for rect in front_faces:
        (x, y, w, h) = rect
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = faceObject(rect)
        frameFacesList.append(face)

    
    tracker.update(frameFacesList)
    registryFacesList = tracker.getFaces(framerate=fps)

    for rect in registryFacesList:
        (x, y, w, h) = rect
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        sub_frame = cv2.GaussianBlur(frame[y:(y+h), x:(x+w)], (27, 27), 30)
        frame[y:(y+h), x:(x+w)] = sub_frame

    del frameFacesList
    del registryFacesList

    #Conta o tempo de fim do ciclo
    tempo_fim = time.time()
    #Calcula FPS (frames por segundo)
    fps = 1/(tempo_fim - tempo_inicio)

    #Coloca o FPS na tela
    cv2.putText(frame, str(fps) ,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,255),2,cv2.LINE_AA)
    #Mostra o frame com eventuais rostos detectados
    cv2.imshow("Frame", frame)
    #Se pressionar Esc ( = key 27), fecha a janela
    key = cv2.waitKey(1)
    if key == 27:
        break

#Libera a camera e fecha as janelas
cap.release()
cv2.destroyAllWindows()
