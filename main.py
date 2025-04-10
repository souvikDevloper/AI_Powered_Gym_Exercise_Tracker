import cv2
import time
import argparse
from tracker import ExerciseTracker, ExerciseType

def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Powered Gym Exercise Tracker: Track push-ups, dumbbell lifts, bicep curls, and tricep extensions using computer vision."
    )
    parser.add_argument(
        '--exercise',
        type=str,
        choices=['pushup', 'dumbbell', 'bicep', 'tricep'],
        default='pushup',
        help="Specify exercise type: pushup, dumbbell, bicep, or tricep (default: pushup)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    exercise_map = {
        'pushup': ExerciseType.PUSHUP,
        'dumbbell': ExerciseType.DUMBBELL,
        'bicep': ExerciseType.BICEP_CURL,
        'tricep': ExerciseType.TRICEP_EXTENSION,
    }
    exercise_type = exercise_map.get(args.exercise, ExerciseType.PUSHUP)
    tracker = ExerciseTracker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print(f"Tracking exercise: {exercise_type.value}")
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame for the chosen exercise
            frame = tracker.process_frame(frame, exercise_type)

            # Display the video feed with overlays
            cv2.imshow('Exercise Tracker', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    elapsed_minutes = (time.time() - start_time) / 60.0
    calories = elapsed_minutes * tracker.calories_per_minute[exercise_type]
    print(f"\nFinal Stats:")
    print(f"Exercise: {exercise_type.value}")
    print(f"Repetitions: {tracker.counter}")
    print(f"Calories Burned: {calories:.2f}")

if __name__ == '__main__':
    main()
