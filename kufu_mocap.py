
import cv2
import mediapipe as mp
import os
import json


REFERENCE_FOLDER = r"C:\Users\chasi\sample photo"

OUTPUT_JSON = os.path.join(REFERENCE_FOLDER,"landmarks_json")

OUTPUT_ANNOTATED = os.path.join(REFERENCE_FOLDER,"landmarks_annotated")

os.makedirs(OUTPUT_JSON, exist_ok=True)
os.makedirs(OUTPUT_ANNOTATED, exist_ok=True)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def save_landmarks(image_path, results, filename):
    """Save detected landmarks into a JSON file."""
    landmarks = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })
    json_path = os.path.join(OUTPUT_JSON, filename + ".json")
    with open(json_path, "w") as f:
        json.dump(landmarks, f, indent=4)

def process_image(image_path):
    print(f"[Info] Processing {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Error] Cannot read {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose:
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            print(f"[Warning] No landmarks detected in {image_path}")
            return

        
        filename = os.path.splitext(os.path.basename(image_path))[0]
        save_landmarks(image_path, results, filename)

        
        annotated = image.copy()
        mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )
        annotated_path = os.path.join(OUTPUT_ANNOTATED, filename + "_landmarked.png")
        cv2.imwrite(annotated_path, annotated)


def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Error] Cannot read {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            print(f"[Warning] No landmarks in {image_path}")
            return

        
        landmarks = [
            {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            }
            for lm in results.pose_landmarks.landmark
        ]

        json_path = os.path.splitext(image_path)[0] + ".json"
        with open(json_path, "w") as f:
            json.dump(landmarks, f, indent=2)

        
        annotated = image.copy()
        mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
        annotated_path = os.path.splitext(image_path)[0] + "_landmarked.png"
        cv2.imwrite(annotated_path, annotated)

        print(f"[Saved] {json_path} and {annotated_path}")       

def main():
    if not os.path.exists(REFERENCE_FOLDER):
        print(f"[Error] Folder not found: {REFERENCE_FOLDER}")
        return

    for file in os.listdir(REFERENCE_FOLDER):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(REFERENCE_FOLDER, file)
            process_image(image_path)

if __name__ == "__main__":
    main()
