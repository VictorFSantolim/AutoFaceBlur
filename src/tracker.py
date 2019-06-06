import math

from kalman_filter import FacePredict

class faceObject:
    'Classe de base para rastrear faces'

    def __init__(self, rect):
        (x, y, w, h) = rect
        self.center = (int(x + w/2.0), int(y + h/2.0))
        self.radius = max(w, h)/2.0
        self.kalman_predictor = FacePredict(0);
    
    def update(self, aface):
        (x, y, w, h) = (0,0,0,0)

        if type(aface) != list:
            (x, y, w, h) = self.kalman_predictor.getKalmanUpdate(aface.getrect())
        else:
            (x, y, w, h) = self.kalman_predictor.getKalmanUpdate(aface)

        self.center = (int(x + w/2.0), int(y + h/2.0))
        self.radius = max(w, h)/2.0

        return self.getrect()

    def getrect(self):
        # Retorna um quadrado que sobrescreve no plano o ciculo definido por center-radius
        return (int(self.center[0]-self.radius), int(self.center[1]-self.radius), int(2.0*self.radius), int(2.0*self.radius))

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
        self.faceMissingCounterReachedMax = dict()
        # Controla os IDs que ja existiram no rastreamento
        self.faceIDCounter = 0

    def update(self, frameFacesList):
        # Cria um dicionario a partir da lista de faces do frame
        frameFacesDict = dict(enumerate(frameFacesList))
        deleteKeyList = list()
        # Associa cada face do registro a alguma(s) face(s) do frame
        for faceID, regFace in self.faceDict.items():
            equalFaces = [ key for key in frameFacesDict if regFace == frameFacesDict[key] ]
            if len(equalFaces) == 0:
                # A face eh considerada perdida caso nao tenha ninguem para associar
                self.faceMissingCounter[faceID] -= 1
                if self.faceMissingCounter[faceID] <= 0:
                    deleteKeyList.append(faceID)
            else:
                # Atualiza baseado no raio mais próximo
                closest = equalFaces[0]
                for key in equalFaces:
                    if abs(self.faceDict[faceID].radius - frameFacesDict[key].radius) < abs(self.faceDict[faceID].radius - frameFacesDict[closest].radius):
                        closest = key
                
                trig_distance = 0.5
                if frameFacesDict[closest].radius < trig_distance*regFace.radius:
                    # Raio muito menor, possivel falso positivo
                    self.faceMissingCounter[faceID] -= 1
                    if self.faceMissingCounter[faceID] <= 0:
                        deleteKeyList.append(faceID)
                else:
                    regFace.update(frameFacesDict[closest])
                    if self.faceMissingCounter[faceID] < self.missingMax:
                        if self.faceMissingCounterReachedMax[faceID]:
                            self.faceMissingCounter[faceID] = self.missingMax
                        else:
                            self.faceMissingCounter[faceID] += 2
                    else:
                        self.faceMissingCounterReachedMax[faceID] = True

                # Remove todas as faces "iguais" do dicionario de faces do frame
                for key in equalFaces:
                    del frameFacesDict[key]

        for key in deleteKeyList:
            del self.faceDict[key]
            del self.faceMissingCounter[key]

        # Para as faces do frame restantes, acrescentar no registro
        for newFaceID in frameFacesDict:
            self.faceDict[self.faceIDCounter] = faceObject(frameFacesDict[newFaceID].getrect())
            self.faceMissingCounter[self.faceIDCounter] = 0
            self.faceMissingCounterReachedMax[self.faceIDCounter] = False
            self.faceIDCounter += 1

        frameFacesDict.clear()
        del frameFacesDict
    
    def getFaces(self, framerate = 30.0):
        # Cria uma lista de retangulos que representam as faces
        faceList = list()
        for faceID in self.faceDict:
            if self.faceMissingCounter[faceID] > 0:
                self.faceDict[faceID].radius = (1.005)*self.faceDict[faceID].radius
                faceList.append(self.faceDict[faceID].update([]))
            else:
                faceList.append(self.faceDict[faceID].getrect())
        return faceList

