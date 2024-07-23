import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
from ultralytics import YOLO
import base64
import numpy as np
import logging
import threading

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO)

# Initialisation de l'application Flask et de SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)

# Création d'un événement pour synchroniser le démarrage de la détection
start_detection_event = threading.Event()

# Chargement du modèle YOLO
model = YOLO(r"C:\Users\ac134\Desktop\Downloads\Detetction jeans\best.pt")

# Chemin de la vidéo
video_path = r"C:\Users\ac134\Desktop\Downloads\Detetction jeans\video.mp4"

# Variable globale pour stocker le nombre total d'objets détectés
total_objects_detected = 0

# Fonction de détection des objets
def detect_objects():
    global total_objects_detected

    # Attendre que l'interface utilisateur soit prête
    start_detection_event.wait()

    try:
        # Ouverture de la capture vidéo
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file at {video_path}")

        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        interval = int(frame_rate * 1.8)
        current_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_frame += 1

            if current_frame % interval == 0:
                results = model.predict(frame)
                num_objects_detected = sum(len(result.boxes) for result in results)
                total_objects_detected += num_objects_detected
                logging.info(f"Frame {current_frame}: {num_objects_detected} objects detected, total: {total_objects_detected}")

                # Émettre la mise à jour au client via SocketIO
                socketio.emit('update_count', {'count': total_objects_detected})

                for result in results:
                    for bbox in result.boxes.xyxy:
                        x1, y1, x2, y2 = bbox[:4]
                        label = f'{result.names[int(bbox[5])]} {bbox[4]:.2f}' if len(bbox) >= 6 else 'Unknown'
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # Encoder la frame en base64 et envoyer au client
                _, buffer = cv2.imencode('.jpg', frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('detection_frame', {'image': frame_b64})

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        logging.error(f"Error in detect_objects: {e}")

# Fonction pour diffuser la vidéo sans détection
def stream_video():
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file at {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        logging.error(f"Error in stream_video: {e}")

@app.route('/')
def index():
    # Signaler que l'interface utilisateur est prête
    start_detection_event.set()
    return render_template('interface.html')

@app.route('/video_feed')
def video_feed():
    return Response(stream_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    emit('connected', {'data': 'Connected to server'})

if __name__ == '__main__':
    # Lancer la détection dans un thread de fond après le démarrage du serveur
    eventlet.spawn(detect_objects)
    socketio.run(app, debug=True)
