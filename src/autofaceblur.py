#Source https://pysource.com/2018/10/01/face-detection-using-haar-cascades-opencv-3-4-with-python-3-tutorial-37/

#Importa as bibliotecas do openCV e do Numpy
import time
import math
import copy

import cv2
import numpy as np




#Captura a imagem da webcam
cap = cv2.VideoCapture(0)
#Importa os modelos xml dos cascades
frontal_face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

class faceObject:
    'Classe de base para rastrear faces'

    history_size_max = 5

    def __init__(self, rect):
        self.history = list()
        (x, y, w, h) = rect
        self.center = (int(x + w/2.0), int(y + h/2.0))
        self.radius = max(w, h)/2.0
        self.history.append(self.center)
    
    def update(self, aface):
        # Atualiza o centro para a nova face
        self.center = aface.center
        # O novo raio eh a media do raio antigo e do raio da nova face
        keep_radius = 0.5
        self.radius = keep_radius*self.radius + (1.0-keep_radius)*aface.radius
        self.history.append(self.center)
        if len(self.history) > faceObject.history_size_max:
            self.history.pop(0)

    def getrect(self):
        # Retorna um quadrado que sobrescreve no plano o ciculo definido por center-radius
        return (int(self.center[0]-self.radius), int(self.center[1]-self.radius), int(2.0*self.radius), int(2.0*self.radius))

    def getpredictcenter(self, n):
        if len(self.history) <= 1:
            return self.center
        else:
            # Previsao linear de n frames futuros baseado no historico
            diffx = (self.history[len(self.history)-1][0] - self.history[0][0])/(len(self.history)-1)
            diffy = (self.history[len(self.history)-1][1] - self.history[0][1])/(len(self.history)-1)
            return (int(n*diffx + self.center[0]), int(n*diffy + self.center[1]))

    def getpredictrect(self, n):
        pcenter = self.getpredictcenter(n)
        return (int(pcenter[0]-self.radius), int(pcenter[1]-self.radius), int(2.0*self.radius), int(2.0*self.radius))

    def __eq__(self, aface):
        # Uma face eh igual se estiver no raio de acao de self
        (x, y) = self.center
        (xa, ya) = aface.center
        alpha_trig_distance = 1.0
        return (math.sqrt((x-xa)**2 + (y-ya)**2) <= alpha_trig_distance*self.radius)


class faceTracker:
    'Classe de rastreamento de faces - baseado em faceObject'

    missingMax = 50

    def __init__(self):
        self.faceDict = dict()
        self.faceMissingCounter = dict()
        # Controla os IDs que ja existiram no rastreamento
        self.faceIDCounter = 0

    def update(self, frameFacesList):
        # Cria um dicionario a partir da lista de faces do frame
        frameFacesDict = dict(enumerate(frameFacesList))
        deleteKeyList = list()
        # Associa cada face do registro a alguma(s) face(s) do frame
        for faceID in self.faceDict:
            # TODO problema quando detecta o olho e nao o rosto
            equalFaces = [ key for key in frameFacesDict if self.faceDict[faceID] == frameFacesDict[key] ]
            print(len(equalFaces))
            if len(equalFaces) == 0:
                # A face eh considerada perdida caso nao tenha ninguem para associar
                self.faceMissingCounter[faceID] = self.faceMissingCounter[faceID] + 1
                if self.faceMissingCounter[faceID] > faceTracker.missingMax:
                    deleteKeyList.append(faceID)
            else:
                self.faceMissingCounter[faceID] = 0
                # Atualiza baseado no raio mais próximo
                closest = equalFaces[0]
                for key in equalFaces:
                    if abs(self.faceDict[faceID].radius - frameFacesDict[key].radius) < abs(self.faceDict[faceID].radius - frameFacesDict[closest].radius):
                        closest = key
                self.faceDict[faceID].update(frameFacesDict[closest])
                # Remove todas as faces "iguais" do dicionario de faces do frame
                for key in equalFaces:
                    del frameFacesDict[key]

        for key in deleteKeyList:
            del self.faceDict[key]
            del self.faceMissingCounter[key]
        # Para as faces do frame restantes, acrescentar no registro
        for newFaceID in frameFacesDict:
            self.faceDict[self.faceIDCounter] = copy.deepcopy(frameFacesDict[newFaceID])
            self.faceMissingCounter[self.faceIDCounter] = 0
            self.faceIDCounter = self.faceIDCounter + 1

        frameFacesDict.clear()
        del frameFacesDict
    
    def getFaces(self):
        # Cria uma lista de retangulos que representam as faces
        faceList = list()
        for faceID in self.faceDict:
            if self.faceMissingCounter[faceID] > 0:
                faceList.append(self.faceDict[faceID].getpredictrect(self.faceMissingCounter[faceID]))
            else:
                faceList.append(self.faceDict[faceID].getrect())
        return faceList


tracker = faceTracker()

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
        #(x, y, w, h) = face.getrect()
        #sub_frame = cv2.GaussianBlur(frame[y:(y+h), x:(x+w)], (27, 27), 30)
        #frame[y:(y+h), x:(x+w)] = sub_frame
        #frame = cv2.circle(frame, face.center, 1, (0, 0, 255), 2)
        #sub_frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #del face
    
    tracker.update(frameFacesList)
    registryFacesList = tracker.getFaces()

    for rect in registryFacesList:
        (x, y, w, h) = rect
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
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
