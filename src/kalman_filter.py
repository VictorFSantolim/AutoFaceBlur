import cv2 as cv
import numpy as np

class FacePredict:
    def __init__(self, max_missing):
        # >>>> Kalman Filter
        self.stateSize = 6
        self.measureSize = 4
        self.contrSize = 0

        self.kalman = cv.KalmanFilter(self.stateSize, self.measureSize, self.contrSize)

        # State matrix:
        # [x,y,v_x,v_y,w,h]
        self.state = np.zeros(self.stateSize, np.float32)

        # Measurement matrix:
        # [z_x,z_y,z_w,z_h]
        self.measures = np.zeros(self.measureSize, np.float32)

        # Transition State Matrix A
        # Note: set dT at each processing step!
        # [ 1 0 dT 0  0 0 ]
        # [ 0 1 0  dT 0 0 ]
        # [ 0 0 1  0  0 0 ]
        # [ 0 0 0  1  0 0 ]
        # [ 0 0 0  0  1 0 ]
        # [ 0 0 0  0  0 1 ]
        self.kalman.transitionMatrix = np.identity(self.stateSize, np.float32)

        # Measure Matrix H
        # [ 1 0 0 0 0 0 ]
        # [ 0 1 0 0 0 0 ]
        # [ 0 0 0 0 1 0 ]
        # [ 0 0 0 0 0 1 ]
        self.kalman.measurementMatrix = np.zeros((self.measureSize, self.stateSize), np.float32)
        self.kalman.measurementMatrix[0][0] = 1.0
        self.kalman.measurementMatrix[1][1] = 1.0
        self.kalman.measurementMatrix[2][4] = 1.0
        self.kalman.measurementMatrix[3][5] = 1.0

        # Process Noise Covariance Matrix Q
        # [ Ex   0   0     0     0    0  ]
        # [ 0    Ey  0     0     0    0  ]
        # [ 0    0   Ev_x  0     0    0  ]
        # [ 0    0   0     Ev_y  0    0  ]
        # [ 0    0   0     0     Ew   0  ]
        # [ 0    0   0     0     0    Eh ]
        self.kalman.processNoiseCov = np.identity(self.stateSize, np.float32)
        self.kalman.processNoiseCov[0][0] = 1.0
        self.kalman.processNoiseCov[1][1] = 1.0
        self.kalman.processNoiseCov[2][2] = 50.0
        self.kalman.processNoiseCov[3][3] = 50.0
        self.kalman.processNoiseCov[4][4] = 1.0
        self.kalman.processNoiseCov[5][5] = 1.0

        # <<<< Kalman Filter

        self.ticks = 0
        self.found = False

        self.notFoundCount = 0
        self.max_missing = max_missing

    def getKalmanUpdate(self, selected_rect):
        previousTick = self.ticks
        self.ticks = cv.getTickCount()

        dT = (self.ticks - previousTick) / cv.getTickFrequency() #seconds
        
        return_rect = selected_rect

        if self.found:
            # >>>> Matrix A
            self.kalman.transitionMatrix[0][2] = dT
            self.kalman.transitionMatrix[1][3] = dT
            # <<<< Matrix A

            self.state = self.kalman.predict()

            predRect = [0, 0, 0, 0]
            predRect[2] = self.state[4]
            predRect[3] = self.state[5]
            predRect[0] = self.state[0] - predRect[2] / 2
            predRect[1] = self.state[1] - predRect[3] / 2

            return_rect = list(predRect)

            center = [0,0]
            center[0] = self.state[0]
            center[1] = self.state[1]

        # >>>>> Kalman Update
        if len(selected_rect) == 0:
            self.notFoundCount += 1
            if self.max_missing != 0:
                if self.notFoundCount >= self.max_missing:
                    self.found = False
        else:
            self.notFoundCount = 0

            self.measures[0] = selected_rect[0] + selected_rect[2] / 2
            self.measures[1] = selected_rect[1] + selected_rect[3] / 2
            self.measures[2] = selected_rect[2]
            self.measures[3] = selected_rect[3]

            if not self.found: # First detection!
                # >>>> Initialization
                self.kalman.errorCovPre[0][0] = 1 # px
                self.kalman.errorCovPre[1][1] = 1 # px
                self.kalman.errorCovPre[2][2] = 1
                self.kalman.errorCovPre[3][3] = 1
                self.kalman.errorCovPre[4][4] = 1 # px
                self.kalman.errorCovPre[5][5] = 1 # px

                self.state[0] = self.measures[0]
                self.state[1] = self.measures[1]
                self.state[2] = 0
                self.state[3] = 0
                self.state[4] = self.measures[2]
                self.state[5] = self.measures[3]
                # <<<< Initialization

                self.kalman.statePost = self.state

                self.found = True
            else:
                self.kalman.correct(self.measures) # Kalman Correction

        return return_rect
        # <<<<< Kalman Update
