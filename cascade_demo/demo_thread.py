# Fonte https://pysource.com/2018/10/01/face-detection-using-haar-cascades-opencv-3-4-with-python-3-tutorial-37/

# Importa as bibliotecas necessarias
import cv2
import numpy as np

from threading import Thread, Lock
import time

# Fonte: https://gist.github.com/allskyee/7749b9318e914ca45eb0a1000a81bf56
# Define uma classe que realiza a leitura da webcam em paralelo (thread)
class WebcamVideoStream :
    # Configuração da classe
    def __init__(self, src = 0) :
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    # Inicialização da classe
    def start(self) :
        if self.started :
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    # Função que realiza a leitura da webcam paralelamente e armazena o frame
    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    # Função chamada para pegar o frame da webcam armazenado
    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    # Termina a thread
    def stop(self) :
        self.started = False
        self.thread.join()

    # Libera a webcam
    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()

# Importa o modelo xml do cascade de face frontal
frontal_face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

# Abre uma janela
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

# Inicializa a captura paralela de frames da webcam
cap = WebcamVideoStream()
cap.start()
fps = 0

while True:
    # Conta o tempo de inicio do ciclo
    tempo_inicio = time.time()
    # Le um frame da camera
    frame = cap.read()
    # Transforma em escala de cinza (melhor para o reconheciento vai haar cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detecta as faces de frente
    front_faces = frontal_face_cascade.detectMultiScale(gray, 1.3, 3)
  
    # Desenha quadrados verdes sobre todas as faces frontais
    for rect in front_faces:
        (x, y, w, h) = rect
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Coloca o FPS na tela
    cv2.putText(frame, str(fps) ,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,255),2,cv2.LINE_AA)

    # Mostra o frame com eventuais rostos detectados e o fps
    cv2.imshow("Frame", frame)
    
    # Se pressionar Esc ( = key 27), fecha a janela
    key = cv2.waitKey(1)
    if key == 27:
        break

    # Conta o tempo de fim do ciclo
    tempo_fim = time.time()
    # Calcula FPS (frames por segundo), para ser exibido no proximo frame
    fps = 1/(tempo_fim - tempo_inicio)
    
# Chama a função de encerrar a captura pela webcam, e fecha as janelas
cap.stop()
cv2.destroyAllWindows()