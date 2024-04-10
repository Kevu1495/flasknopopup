from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import math
import threading

app = Flask(__name__)

frame_width = 640
frame_height = 480
frame_rate = 60

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize VideoCapture object
cap = cv2.VideoCapture(0)
def evaluate_bicep_curl(landmarks):
    # Extract relevant landmarks for bicep curl evaluation
    left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
    left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]

    right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_elbow = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]
    right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]

    # Calculate angles for left arm
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    # Calculate angles for right arm
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Assume correct bicep curl angle is around 90 degrees
    bicep_curl_threshold = 20  # Adjust as needed

    # Check if both arms have correct angles
    if left_arm_angle < bicep_curl_threshold and right_arm_angle < bicep_curl_threshold:
        return True
    else:
        return False


def calculate_angle(a, b, c):
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle = math.degrees(radians)
    return angle


def detect_pose_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = {lmk: lm for lmk, lm in enumerate(results.pose_landmarks.landmark)}
        is_bicep_curl = evaluate_bicep_curl(landmarks)

        # Draw feedback on frame
        if is_bicep_curl:
            cv2.putText(frame, "Bicep Curl Correct", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Incorrect Form", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        relevant_landmarks = [mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                              mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
                              mp.solutions.pose.PoseLandmark.LEFT_WRIST,
                              mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
                              mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
                              mp.solutions.pose.PoseLandmark.RIGHT_WRIST]

        for landmark in relevant_landmarks:
            landmark_point = results.pose_landmarks.landmark[landmark.value]
            h, w, c = frame.shape
            cx, cy = int(landmark_point.x * w), int(landmark_point.y * h)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # Draw a circle f

    return frame


def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = detect_pose_landmarks(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/exercise1')
def exercise1():
    return render_template('exercise1.html')

@app.route('/exercise2')
def exercise2():
    return render_template('exercise2.html')


@app.route('/video_feed')
def video_feed():
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, frame_rate)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
