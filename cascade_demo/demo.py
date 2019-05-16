# Source https://pysource.com/2018/10/01/face-detection-using-haar-cascades-opencv-3-4-with-python-3-tutorial-37/

# Importa as bibliotecas do openCV e do Numpy
import cv2
import numpy as np

import time
import threading

# Source http://blog.blitzblit.com/2017/12/24/asynchronous-video-capture-in-python-with-opencv/
class VideoCaptureAsync:
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Asynchroneous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()

def main():
    # Captura a imagem da webcam
    cap = VideoCaptureAsync(0)
    # Importa os modelos xml dos cascades
    frontal_face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    profile_face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_profileface.xml")

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

    cap.start()
    while True:
        # Conta o tempo de inicio do ciclo
        tempo_inicio = time.time()
        # Le um frame da camera
        ret , frame = cap.read()
        # Transforma em escala de cinza (melhor para o reconheciento vai haar cascade)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detecta as faces de frente e de perfil
        front_faces = frontal_face_cascade.detectMultiScale(gray, 1.3, 3)
        profile_faces = profile_face_cascade.detectMultiScale(gray, 1.3, 3)

        # Desenha quadrados verdes dobre todas as faces frontais
        for rect in front_faces:
            (x, y, w, h) = rect
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Desenha quadrados azuis sobre todas as faces de perfil
        for rect in profile_faces:
             (x, y, w, h) = rect
             frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # Conta o tempo de fim do ciclo
        tempo_fim = time.time()
        # Calcula FPS (frames por segundo)
        fps = 1/(tempo_fim - tempo_inicio)

        # Coloca o FPS na tela
        cv2.putText(frame, str(fps) ,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,255),2,cv2.LINE_AA)
        # Mostra o frame com eventuais rostos detectados
        cv2.imshow("Frame", frame)
        # Se pressionar Esc ( = key 27), fecha a janela
        key = cv2.waitKey(1)
        if key == 27:
            break
        
    # Libera a camera e fecha as janelas
    cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
