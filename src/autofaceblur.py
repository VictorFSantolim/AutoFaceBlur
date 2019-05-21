#Source https://pysource.com/2018/10/01/face-detection-using-haar-cascades-opencv-3-4-with-python-3-tutorial-37/

#Importa as bibliotecas do openCV e do Numpy
import time
import os

import cv2
import numpy as np
import math



#Captura a imagem da webcam
cap = cv2.VideoCapture(0)
#Importa os modelos xml dos cascades
frontal_face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

class faceObject:
    'Classe de base para rastrear faces'

    history_size_max = 5
    alpha_trig_distance = 1.0

    def __init__(self, rect):
        self.history = list()
        (x, y, w, h) = rect
        self.center = (int(x + w/2.0), int(y + h/2.0))
        self.radius = max(w, h)
        self.history.append(self.center)
    
    def update(self, rect):
        self.history.append(rect)
        (x, y, w, h) = rect
        self.center = (x + w/2.0, y + h/2.0)
        self.radius = max(w, h)
        self.history.append(self.center)
        if len(self.history) > faceObject.history_size_max:
            self.history.pop(0)

    def getrect(self):
        return (int(self.center[0]-self.radius/2), int(self.center[1]-self.radius/2), self.radius, self.radius)

    def __eq__(self, aface):
        (x, y) = self.center
        (xa, ya) = aface.center
        return (math.sqrt((x-xa)**2 + (y-ya)**2) <= faceObject.alpha_trig_distance*self.radius)


while True:
    #Conta o tempo de inicio do ciclo
    tempo_inicio = time.time()
    #Le um frame da camera
    ret , frame = cap.read()
    #Transforma em escala de cinza (melhor para o reconhecimento vai haar cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Detecta as faces de frente e de perfil
    front_faces = frontal_face_cascade.detectMultiScale(gray, 1.3, 3)


    for rect in front_faces:
        (x, y, w, h) = rect
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        sub_frame = cv2.GaussianBlur(frame[y:(y+h), x:(x+w)], (23, 23), 30)
        frame[y:(y+h), x:(x+w)] = sub_frame
        face = faceObject(rect)
        frame = cv2.circle(frame, face.center, 1, (0, 0, 255), 2)
        del face
    
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
