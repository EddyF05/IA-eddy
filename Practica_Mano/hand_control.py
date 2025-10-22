# hand_control.py
import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
center = (320, 240)  # cambiar según resolución
while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    center = (w//2, h//2)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    radius = 50  # valor por defecto
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            # landmarks: 4 = tip pulgar, 8 = tip indice
            x1 = int(handLms.landmark[4].x * w)
            y1 = int(handLms.landmark[4].y * h)
            x2 = int(handLms.landmark[8].x * w)
            y2 = int(handLms.landmark[8].y * h)
            dist = math.hypot(x2-x1, y2-y1)
            # mapear distancia a radio (ajusta factores)
            radius = int(min(max(dist, 20), 300) // 2)
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
    # dibujar figura central (ejemplo: círculo)
    cv2.circle(frame, center, radius, (0,255,0), -1)
    cv2.putText(frame, f"Radius: {radius}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Hand Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
