import cv2
import time
import numpy as np

# Load cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera Not Accessible")
    exit()

print("Exam Proctoring Started. Press ESC to exit.")

# ---------------- CONFIG ----------------
NO_FACE_TIME_LIMIT = 2.0        # seconds
MULTI_FACE_TIME_LIMIT = 2.0
LOOK_AWAY_TIME_LIMIT = 2.0
EYES_MISSING_TIME_LIMIT = 2.0

CENTER_THRESHOLD = 150           # pixels
# ---------------------------------------

# Timers
no_face_start = None
multi_face_start = None
look_away_start = None
eyes_missing_start = None

# Violation counters (event-based)
no_face_events = 0
multi_face_events = 0
look_away_events = 0
suspicious_events = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    status = "Normal"
    color = (0, 255, 0)
    current_time = time.time()

    # ---------------- NO FACE ----------------
    if len(faces) == 0:
        if no_face_start is None:
            no_face_start = current_time
        elif current_time - no_face_start >= NO_FACE_TIME_LIMIT:
            status = "No Face Detected"
            color = (0, 0, 255)
    else:
        if no_face_start is not None:
            if current_time - no_face_start >= NO_FACE_TIME_LIMIT:
                no_face_events += 1
        no_face_start = None

    # ---------------- MULTIPLE FACES ----------------
    if len(faces) > 1:
        if multi_face_start is None:
            multi_face_start = current_time
        elif current_time - multi_face_start >= MULTI_FACE_TIME_LIMIT:
            status = "Multiple Faces Detected"
            color = (0, 0, 255)
    else:
        if multi_face_start is not None:
            if current_time - multi_face_start >= MULTI_FACE_TIME_LIMIT:
                multi_face_events += 1
        multi_face_start = None

    # ---------------- SINGLE FACE LOGIC ----------------
    if len(faces) == 1:
        x, y, w, h = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        face_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_gray, 1.3, 5)

        # -------- Eyes missing (suspicious, not cheating) --------
        if len(eyes) == 0:
            if eyes_missing_start is None:
                eyes_missing_start = current_time
            elif current_time - eyes_missing_start >= EYES_MISSING_TIME_LIMIT:
                status = "Eyes Not Visible"
                color = (0, 165, 255)
        else:
            if eyes_missing_start is not None:
                if current_time - eyes_missing_start >= EYES_MISSING_TIME_LIMIT:
                    suspicious_events += 1
            eyes_missing_start = None

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                frame,
                (x + ex, y + ey),
                (x + ex + ew, y + ey + eh),
                (255, 0, 0),
                2
            )

        # -------- Looking Away (head position) --------
        face_center = x + w // 2
        frame_center = frame.shape[1] // 2

        if abs(face_center - frame_center) > CENTER_THRESHOLD:
            if look_away_start is None:
                look_away_start = current_time
            elif current_time - look_away_start >= LOOK_AWAY_TIME_LIMIT:
                status = "Looking Away"
                color = (0, 165, 255)
        else:
            if look_away_start is not None:
                if current_time - look_away_start >= LOOK_AWAY_TIME_LIMIT:
                    look_away_events += 1
            look_away_start = None

    # ---------------- DISPLAY ----------------
    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Exam Detection System", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ---------------- SUMMARY ----------------
print("\n----- Exam Monitoring Summary -----")
print(f"No Face Events        : {no_face_events}")
print(f"Multiple Face Events  : {multi_face_events}")
print(f"Looking Away Events   : {look_away_events}")
print(f"Suspicious Events     : {suspicious_events}")

