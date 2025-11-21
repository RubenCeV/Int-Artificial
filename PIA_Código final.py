#!/usr/bin/env python3
import time
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# configuración de pines
IN1 = 6
IN2 = 5
ENA = 13
PIR_PIN = 17
RELAY_PIN = 22   # LOW = activo

# tiempos del sistema
T_ACTUADOR = 3.0
T_LOCK = 1.0

# parámetros de IA
UMBRAL_CERRAR = 0.76
UMBRAL_ABRIR  = 0.74

MODEL_PATH = "modelo_reconocimiento_facial2.tflite"
HAAR_PATH = "haarcascade_frontalface_default.xml"
CLASES = ["Nadia", "Ruben", "Tadeo"]

TIEMPO_DETECCION_REAL = 2.0

# inicializar GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(PIR_PIN, GPIO.IN)
GPIO.setup(RELAY_PIN, GPIO.OUT)

GPIO.output(IN1, GPIO.LOW)
GPIO.output(IN2, GPIO.LOW)
GPIO.output(ENA, GPIO.LOW)
GPIO.output(RELAY_PIN, GPIO.HIGH)   # cerradura cerrada


# funciones del actuador
def cerrar_cerradura():
    GPIO.output(RELAY_PIN, GPIO.HIGH)

def abrir_cerradura():
    GPIO.output(RELAY_PIN, GPIO.LOW)
    time.sleep(T_LOCK)
    GPIO.output(RELAY_PIN, GPIO.HIGH)

def abrir_puerta():
    print(">> abriendo puerta")
    abrir_cerradura()
    time.sleep(0.2)

    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(ENA, GPIO.HIGH)

    time.sleep(T_ACTUADOR - 1.0)
    time.sleep(1.0)

    GPIO.output(ENA, GPIO.LOW)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)

def cerrar_puerta():
    print(">> cerrando puerta")
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(ENA, GPIO.HIGH)

    time.sleep(T_ACTUADOR - 1.0)
    abrir_cerradura()
    time.sleep(1.0)

    GPIO.output(ENA, GPIO.LOW)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    cerrar_cerradura()


# cargar modelo tflite
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_H, IMG_W = input_details[0]["shape"][1:3]

# detector Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + HAAR_PATH)

# configurar cámara
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(1)


# funciones IA
def recortar_y_preparar(frame):
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return None, None

    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

    pad = int(0.05 * w)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)

    rostro = frame[y1:y2, x1:x2]
    rostro = cv2.resize(rostro, (IMG_W, IMG_H))
    rostro = rostro.astype(np.float32) / 255.0
    rostro = np.expand_dims(rostro, axis=0)

    return rostro, (x1, y1, x2, y2)

def inferir(t):
    interpreter.set_tensor(input_details[0]["index"], t)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0]


# loop principal
print("sistema iniciado con PIR...")

try:
    while True:
        if GPIO.input(PIR_PIN) == GPIO.HIGH:
            inicio = time.time()

            while GPIO.input(PIR_PIN) == GPIO.HIGH:
                if time.time() - inicio >= TIEMPO_DETECCION_REAL:

                    frame = picam2.capture_array()
                    prep, _ = recortar_y_preparar(frame)

                    if prep is None:
                        print("no se detectó rostro.")
                        break

                    probs = inferir(prep)
                    idx = int(np.argmax(probs))
                    prob = float(probs[idx])
                    nombre = CLASES[idx] if idx < len(CLASES) else "ClaseX"

                    print(f">> IA detecta {nombre} ({prob*100:.1f}%)")

                    if prob >= UMBRAL_CERRAR:
                        cerrar_puerta()
                    elif prob <= UMBRAL_ABRIR:
                        abrir_puerta()
                    else:
                        print("zona gris (74–76%), no hacer nada")

                    time.sleep(1)
                    break

                time.sleep(0.05)

        time.sleep(0.05)

except KeyboardInterrupt:
    print("detenido por usuario.")

finally:
    picam2.stop()
    GPIO.cleanup()
    print("limpieza finalizada.")
