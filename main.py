from math import radians

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math

from numpy.ma.core import indices

# Definir donde estan los archivos
ubicacion_modelo_manos = python.BaseOptions(model_asset_path='hand_landmarker.task')
ubicacion_modelo_cara  = python.BaseOptions(model_asset_path='face_landmarker.task')

# Configurar el detector manos y cara
opciones_detector_manos = vision.HandLandmarkerOptions(
    base_options=ubicacion_modelo_manos,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence= 0.9
)

opciones_detector_cara = vision.FaceLandmarkerOptions(
    base_options= ubicacion_modelo_cara,
    running_mode=vision.RunningMode.VIDEO,
    min_face_detection_confidence=0.9
)
# Cargar detector de manos y cara con sus opciones
detectar_manos = vision.HandLandmarker.create_from_options(opciones_detector_manos)
detectar_cara  = vision.FaceLandmarker.create_from_options(opciones_detector_cara)

camara= cv2.VideoCapture(0) #abrir camara

class Funciones_cara:

    def __init__(self, landmarks):
        self.landmarks = landmarks

    def calcular_distancia(self, punto1, punto2):
        punto1 = self.landmarks[punto1]
        punto2 = self.landmarks[punto2]
        return math.sqrt((punto1.x - punto2.x)**2 + (punto1.y - punto2.y)**2 )

    def ancho_boca(self):
        return self.calcular_distancia(13, 14)

    def comisuras_mejillas(self):
        return self.calcular_distancia(61, 291)

    def distancia_esquinas_ojos(self):
        return self.calcular_distancia(33, 263)

    def es_sonrisa(self):
        Radio = self.comisuras_mejillas() /  self.distancia_esquinas_ojos()
        if (self.comisuras_mejillas() > Radio):
            return True




class Funciones_mano:

    def __init__(self, landmark):
        self.landmark = landmark

    def dedo_extendido(self, punta, base):
        punta = self.landmark[punta]
        base  = self.landmark[base]
        return punta.y < base.y

    def obtener_estado_dedos(self):
        indice_extendido = self.dedo_extendido(8, 5)
        medio_extendido = self.dedo_extendido(12, 9)
        anular_extendido = self.dedo_extendido(16, 13)
        meñique_extendido = self.dedo_extendido(20, 17)
        return [indice_extendido, medio_extendido, anular_extendido, meñique_extendido]


    def es_puño(self):
        return not any(self.obtener_estado_dedos())


    def es_mano_abierta(self):
        return all(self.obtener_estado_dedos())


    def es_simbolo_paz(self):
        indice, medio, anular, meñique = self.obtener_estado_dedos()
        dedos_doblados = [not d for d in self.obtener_estado_dedos()]
        anular_doblado = dedos_doblados[2]
        meñique_doblado = dedos_doblados[3]
        return indice and medio and anular_doblado and meñique_doblado


    def es_pulgar_arriba(self):
        pulgar_extendido = self.dedo_extendido(4, 0)
        return pulgar_extendido and not any(self.obtener_estado_dedos())

while True:
    exitoso, frame = camara.read()
    if not exitoso:
        print("Nose pudo acceder a la camara")
        break

    timestamp = int(time.time() * 1000) #obtener el tiempo actual en milisegundos

    # frame_rgb es una matriz de numpy
    #cambiamos el formato de color acorde a mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # mediapipe necesita su propio "frame" por eso lo creamos
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    resultado_manos = detectar_manos.detect_for_video(mp_image, timestamp)
    resultado_cara  = detectar_cara.detect_for_video(mp_image, timestamp)

    # logica para dibujar los puntos en las manos
    mis_manos = resultado_manos.hand_landmarks
    mi_cara = resultado_cara.face_landmarks

    if mis_manos:
        for mano in mis_manos:
            for punto in mano:
                # creamos una trupla de 3 valores y la vez la desempaquetamos
                alto, ancho, _ = frame.shape
                # convertimos los puntos o landmarks a pixeles
                x = int(punto.x * ancho)
                y = int(punto.y * alto)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            # creamos un circulo para cubrir el area
            # colocamos el frame, las coordenas del centro(x,y), el radio, el color y grosor del borde
            # sera de color verde
        #def es_puño(mano):

        mi_mano = Funciones_mano(mano)

        #print(f"Pulgar puntay:", mano[4].y, "| Pulgar base y:", mano[1].y, "| Muñeca y:", mano[0].y)


        if mi_mano.es_mano_abierta():
            print("Es mano Abierta")
        elif mi_mano.es_pulgar_arriba():
            print("Pulgar Hacia Arriba")
        elif mi_mano.es_puño():
            print("Es puño")
        elif mi_mano.es_simbolo_paz():
            print("Paz")







    if mi_cara:
        for cara in mi_cara:
            for punto in cara:
                alto, ancho, _ = frame.shape
                x = int(punto.x * ancho)
                y = int(punto.y * alto)
                cv2.circle(frame, (x,y), 1, (255,0,0), -1)
                # el circulo sera de color azul

        mi_carita = Funciones_cara(cara)

        if mi_carita.es_sonrisa():
            print("Sonrisa :D")



    #abrir ventana y mostrar el fotograma
    cv2.imshow("Mi camara", frame)

    # presionar tecla para salir
    # 1 es tiempo en ms, osea: 1ms
    # ord es el codigo numerico de la tecla
    if cv2.waitKey(1) == ord("q"):
        break
camara.release()
cv2.destroyAllWindows()

