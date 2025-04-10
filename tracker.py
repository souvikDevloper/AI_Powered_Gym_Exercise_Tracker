import cv2
import mediapipe as mp
import numpy as np
import time
from enum import Enum

class ExerciseType(Enum):
    PUSHUP = "Push-up"
    DUMBBELL = "Dumbbell Lift"
    BICEP_CURL = "Bicep Curl"
    TRICEP_EXTENSION = "Tricep Extension"

class ExerciseTracker:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        # Tracking variables
        self.counter = 0
        self.stage = None
        self.start_time = time.time()
        self.calories_per_minute = {
            ExerciseType.PUSHUP: 8,         # calories burned per minute
            ExerciseType.DUMBBELL: 6,
            ExerciseType.BICEP_CURL: 5,
            ExerciseType.TRICEP_EXTENSION: 5
        }

    def calculate_angle(self, a, b, c):
        """Calculate angle (in degrees) between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180:
            angle = 360 - angle
        return angle

    def process_frame(self, frame, exercise_type):
        # Convert frame for processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # Draw landmarks on the frame
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            landmarks = results.pose_landmarks.landmark

            # Process tracking based on chosen exercise type
            if exercise_type == ExerciseType.PUSHUP:
                self.track_pushup(landmarks)
            elif exercise_type == ExerciseType.DUMBBELL:
                self.track_dumbbell(landmarks)
            elif exercise_type == ExerciseType.BICEP_CURL:
                self.track_bicep_curl(landmarks)
            elif exercise_type == ExerciseType.TRICEP_EXTENSION:
                self.track_tricep_extension(landmarks)

        # Overlay the stats on the image
        self.update_display(image, exercise_type)
        return image

    def track_pushup(self, landmarks):
        """Track push-up movement and count repetitions."""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        angle = self.calculate_angle(shoulder, elbow, wrist)

        if angle > 160:
            self.stage = "up"
        if angle < 90 and self.stage == "up":
            self.stage = "down"
            self.counter += 1

    def track_dumbbell(self, landmarks):
        """Track dumbbell lift movement and count repetitions."""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        angle = self.calculate_angle(shoulder, elbow, wrist)

        if angle < 30:
            self.stage = "down"
        if angle > 150 and self.stage == "down":
            self.stage = "up"
            self.counter += 1

    def track_bicep_curl(self, landmarks):
        """Track bicep curl movement and count repetitions."""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        angle = self.calculate_angle(shoulder, elbow, wrist)

        if angle > 160:
            self.stage = "down"
        if angle < 50 and self.stage == "down":
            self.stage = "up"
            self.counter += 1

    def track_tricep_extension(self, landmarks):
        """Track tricep extension movement and count repetitions."""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        angle = self.calculate_angle(shoulder, elbow, wrist)

        if angle < 30:
            self.stage = "up"
        if angle > 150 and self.stage == "up":
            self.stage = "down"
            self.counter += 1

    def update_display(self, image, exercise_type):
        """Overlay exercise information on the video feed."""
        cv2.putText(
            image, f'Exercise: {exercise_type.value}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        )
        cv2.putText(
            image, f'Reps: {self.counter}', (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        )
        if exercise_type == ExerciseType.DUMBBELL:
            total_weight = self.counter * 10  # Assume dumbbell weight is 10 kg per rep
            cv2.putText(
                image, f'Total Weight: {total_weight} kg', (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
            )
        cv2.putText(
            image, f'Stage: {self.stage if self.stage else "N/A"}', (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        )
