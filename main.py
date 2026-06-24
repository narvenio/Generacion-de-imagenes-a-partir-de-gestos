from math import radians

import cv2
from math import radians

import cv2
import mediapipe as mp
from PIL.ImageChops import overlay
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math

from numpy.ma.core import indices

#definir variables para calibracion

calibrado = False
muestras_calibracion = []
tiempo_inicio = time.time()
fps_anterior = time.time()
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

    def radio_boca(self):
        radio = self.comisuras_mejillas() /  self.distancia_esquinas_ojos()
        return radio

    def es_sonrisa(selfs, umbral_sonrisa):
        if (selfs.radio_boca() > umbral_sonrisa):
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

        if not calibrado:
            alto, ancho, _ = frame.shape


            # Crear el fondo antes del texto
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (ancho, alto), (200, 200, 200), -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7,0)

            # Crear texto "calibrando"..
            texto = "***Calibrando...pon un rostro neutral***"
            (ancho_texto, _), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            x_centrado = (ancho - ancho_texto) // 2
            y_centrado = (alto // 2)

            cv2.putText(
                frame,
                texto,
                (x_centrado, y_centrado),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0,0,0),
                2
            )

            tiempo_actual = time.time()
            segundos_pasados = tiempo_actual - tiempo_inicio

            # Crear Barra de proceso
            progreso = min(segundos_pasados / 5, 1.0)

            # Divides los segundos pasados / 3 porque son 3 segundos y tomas el valor mas pequeño hasta que llegue a 1.0, con eso nos aseguramos
            ancho_barra = int(ancho * progreso)
            cv2.rectangle(frame, (0, alto - 20), (ancho_barra, alto), (0,255,0), -1)

            ratio_boca = mi_carita.radio_boca()
            muestras_calibracion.append(ratio_boca)

            if segundos_pasados > 5:
                sumatoria_de_muestras = sum(muestras_calibracion)
                promedio_muestras = sumatoria_de_muestras / len(muestras_calibracion)
                ratio_neutral = promedio_muestras
                umbral_sonrisa  = ratio_neutral * 1.25 # añadimos un 15% como forma de seguridad para estar seguros del ratio
                calibrado = True
        else:
             if mi_carita.es_sonrisa(umbral_sonrisa):
                print("Sonrisa :D")

    fps_actual = time.time()
    if fps_actual - fps_anterior > 0:
        fps = 1 / (fps_actual - fps_anterior)

        cv2.putText(frame, f"FPS: {int(fps)}",
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0),2
                    )
    fps_anterior = fps_actual

    #abrir ventana y mostrar el fotograma
    cv2.imshow("Mi camara", frame)