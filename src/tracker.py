import math


class faceObject:
    'Class for a face tracking'

    history_size_max = 50

    def __init__(self, rect):
        self.history = list()
        (x, y, w, h) = rect
        self.center = (int(x + w/2.0), int(y + h/2.0))
        self.radius = max(w, h)/2.0
        self.history.append(self.center)

    def update(self, aface):
        # Updates de center based on a face center
        self.center = aface.center
        # New radius is the pounded average of old radius and a face radius
        keep_radius = 0.64
        greater_radius = max(self.radius, aface.radius)
        minor_radius = min(self.radius, aface.radius)
        self.radius = keep_radius*greater_radius + (1.0-keep_radius)*minor_radius
        self.history.append(self.center)
        if len(self.history) > faceObject.history_size_max:
            self.history.pop(0)

    def getrect(self):
        # Returns a square that overrides the (center, radius) circle of this face
        return (int(self.center[0]-self.radius), int(self.center[1]-self.radius), int(2.0*self.radius), int(2.0*self.radius))

    def getpredictcenter(self, n):
        if len(self.history) <= 1:
            return self.center
        else:
            # Linear prediction based on center history n frames ahead
            diffx = (self.history[len(self.history)-1][0] -
                     self.history[0][0])/(len(self.history)-1)
            diffy = (self.history[len(self.history)-1][1] -
                     self.history[0][1])/(len(self.history)-1)
            return (int(n*diffx + self.center[0]), int(n*diffy + self.center[1]))

    def getpredictrect(self, n):
        pcenter = self.getpredictcenter(n)
        return (int(pcenter[0]-self.radius), int(pcenter[1]-self.radius), int(2.0*self.radius), int(2.0*self.radius))

    def __eq__(self, aface):
        # Consider a face equal to this face if the center is on this face radius
        (x, y) = self.center
        (xa, ya) = aface.center
        alpha_trig_distance = 1.0
        return (math.sqrt((x-xa)**2 + (y-ya)**2) <= alpha_trig_distance*self.radius)


class faceTracker:
    'Face tracking class based on faceObject'

    missingMax = 50

    def __init__(self):
        self.faceDict = dict()
        self.faceMissingCounter = dict()
        # Represents all the ID's already tracked
        self.faceIDCounter = 0

    def update(self, frameFacesList):
        # Creates a dictionary based on faces list of the frame
        frameFacesDict = dict(enumerate(frameFacesList))
        deleteKeyList = list()
        # Creates a relationship between faces on the registry
        # and faces on the frame
        for faceID, regFace in self.faceDict.items():
            equalFaces = [key for key in frameFacesDict if regFace == frameFacesDict[key]]
            if len(equalFaces) == 0:
                # Consider missing face if no face on the frame id equal to this face
                self.faceMissingCounter[faceID] += 1
                if self.faceMissingCounter[faceID] > faceTracker.missingMax or self.faceMissingCounter[faceID] > 2*len(regFace.history):
                    deleteKeyList.append(faceID)
            else:
                # Update based on the closest face on equalFaces list
                closest = equalFaces[0]
                for key in equalFaces:
                    if abs(self.faceDict[faceID].radius - frameFacesDict[key].radius) < abs(self.faceDict[faceID].radius - frameFacesDict[closest].radius):
                        closest = key

                trig_distance = 0.5
                if frameFacesDict[closest].radius < trig_distance*regFace.radius:
                    # If closest face radius is too little, it does not update
                    self.faceMissingCounter[faceID] += 1
                    if self.faceMissingCounter[faceID] > faceTracker.missingMax:
                        deleteKeyList.append(faceID)
                else:
                    regFace.update(frameFacesDict[closest])
                    self.faceMissingCounter[faceID] -= 10
                    if self.faceMissingCounter[faceID] < 0:
                        self.faceMissingCounter[faceID] = 0

                # Remove all equalFaces on dictionary
                for key in equalFaces:
                    del frameFacesDict[key]

        for key in deleteKeyList:
            del self.faceDict[key]
            del self.faceMissingCounter[key]

        # The left faces are new on the registry
        for newFaceID in frameFacesDict:
            self.faceDict[self.faceIDCounter] = faceObject(
                frameFacesDict[newFaceID].getrect())
            self.faceMissingCounter[self.faceIDCounter] = 0
            self.faceIDCounter += 1

        frameFacesDict.clear()
        del frameFacesDict

    def getFaces(self, framerate=30.0):
        # Create a list of squares around the faces
        faceList = list()
        for faceID in self.faceDict:
            if self.faceMissingCounter[faceID] > 0:
                framerate_div = framerate/30.0

                self.faceDict[faceID].radius = (1.005)*self.faceDict[faceID].radius
                faceList.append(self.faceDict[faceID].getpredictrect(
                    self.faceMissingCounter[faceID]/framerate_div))
            else:
                faceList.append(self.faceDict[faceID].getrect())
        return faceList
