
import os
os.environ["MEDIAPIPE_DISABLE_XNNPACK"] = "1"
import glob
import json
import cv2
import numpy as np
import mediapipe as mp


REFERENCE_DIR = r"C:\Users\chasi\sample photo"   
WIN_NAME = "Kufu Pose Game"
MATCH_THRESHOLD = 0.92            
TEXT_COLOR = (30, 255, 30)


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose



def dicts_to_array(landmarks):
    """Convert Mediapipe landmarks, JSON dicts, or numpy arrays into numpy float32 (33,3)."""
    import numpy as np
    from collections.abc import Iterable

    
    if isinstance(landmarks, np.ndarray):
        return landmarks.astype(np.float32)

  
    if isinstance(landmarks, list) and isinstance(landmarks[0], dict):
        return np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks], dtype=np.float32)

    
    if isinstance(landmarks, Iterable) and hasattr(landmarks[0], "x"):
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    raise ValueError(f"Unsupported landmark format: {type(landmarks)}")


def normalize_landmarks(landmarks):
    """
    Normalize by:
      - translate hip center to origin
      - scale by distance between shoulders
    """
    L = dicts_to_array(landmarks)   
    if L.shape[0] < 25:
       
        return None

    
    hip_center = (L[23] + L[24]) / 2.0
    L = L - hip_center

   
    shoulder_dist = np.linalg.norm(L[11] - L[12]) + 1e-6
    L = L / shoulder_dist

    return L


def cosine_sim(a, b):
    """
    Cosine similarity over all joints (3D flattened).
    Returns scalar in [0,1] (clipped).
    """
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denom = (np.linalg.norm(a_flat) * np.linalg.norm(b_flat)) + 1e-8
    val = float(np.dot(a_flat, b_flat) / denom)
    return max(0.0, min(1.0, val))


def compare_landmarks(user, reference):
    """
    Returns a similarity score (0..1).
    """
    user_norm = normalize_landmarks(user)
    ref_norm = normalize_landmarks(reference)
    if user_norm is None or ref_norm is None:
        return 0.0
    # Use only first 33 joints (defensive)
    user_norm = user_norm[:33]
    ref_norm  = ref_norm[:33]
    return cosine_sim(user_norm, ref_norm)


def load_references():
    """
    Load all *.json landmark files from REFERENCE_DIR.
    Each JSON is expected to be a list of dicts: [{"x":..,"y":..,"z":..}, ...] length 33.
    Returns list of (name, np.ndarray(33,3)).
    """
    refs = []
    json_files = sorted(glob.glob(os.path.join(REFERENCE_DIR, "*.json")))
    if not json_files:
        print(f"[Error] No JSON references found in: {REFERENCE_DIR}")
        return refs

    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            # accept either top-level list or {"landmarks":[...]}
            if isinstance(data, dict) and "landmarks" in data:
                data = data["landmarks"]
            arr = dicts_to_array(data)   # (33,3)
            refs.append((os.path.basename(fp), arr))
            print(f"[Ref] Loaded {fp}")
        except Exception as e:
            print(f"[Warning] Failed to load {fp}: {e}")
    return refs


def grade_from_matches(n_match, total):
    if total == 0:
        return "F"
    if n_match == 0:
        return "F"
    if n_match < 6:
        return "E"
    if n_match == 6:
        return "D"
    if 7 <= n_match <= 8:
        return "C"
    if 9 <= n_match <= 10:
        return "B"
    if n_match >= 11:
        return "A"
    return "E"



def evaluate_session(frames, references, pose):
    """
    Go through recorded frames, find best match per frame, and count
    how many distinct reference poses were matched above threshold.
    We count a ref pose at most once.
    """
    print("[Game] Processing session results...")

    matched_flags = {name: False for name, _ in references}

    for frame in frames:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            continue

        user_L = results.pose_landmarks.landmark

       
        best_name = None
        best_score = 0.0
        for name, ref_L in references:
            score = compare_landmarks(user_L, ref_L)
            if score > best_score:
                best_score = score
                best_name = name

        
        if best_name is not None and best_score >= MATCH_THRESHOLD:
            matched_flags[best_name] = True

    n_matched = sum(1 for v in matched_flags.values() if v)
    total_refs = len(references)
    grade = grade_from_matches(n_matched, total_refs)
    if grade == "F":
        feedback = "You failed,press Esc to exit and try again"
    elif grade == "D":
        feedback = "You pass ,press Esc to exit"
    elif grade == "C":
        feedback = "Well done,you pass, press Esc to exit"
    elif grade == "B":
        feedback = "Excellent,great job, press Esc to exit"
    elif grade == "A":
        feedback = "Perfect,match all steps!,press Esc to exit"
    else:
        feedback = "almost pass,press Esc to exit"
    print(f"[Game] Matched {n_matched}/{total_refs} reference poses")
    print(f"[Game] Final Grade: {grade}")
    print("[Game] " + feedback)
    return n_matched, total_refs, grade, feedback



def main():
    references = load_references()
    if not references:
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Cannot open webcam")
        return

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    recording = False
    recorded_frames = []

    
    with mp_pose.Pose(
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        print("Press S to Start, E to Stop & Evaluate, Q to Quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Error] Failed to read camera")
                break

            
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            
            h, w = frame.shape[:2]
            cv2.putText(frame, "S = Start | E = Stop and Get result | Q = Quit",
                        (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if recording:
                cv2.putText(frame, "REC", (w - 100, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
                recorded_frames.append(frame.copy())

            cv2.imshow(WIN_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('s'), ord('S')):
                print("[Game] Starting after 5 second")
                
                h,w = frame.shape[:2]

                for i in range(5,0,-1):
                    countdown_frame = np.zeros((h, w, 3), dtype=np.uint8)
                    

                    text= f"{i}"
                    font=cv2.FONT_HERSHEY_SIMPLEX
                    font_scale=5
                    thickness=8
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

                    text_x = (w - text_size[0]) // 2
                    text_y = (h + text_size[1]) // 2

                    cv2.putText(countdown_frame, text, (text_x, text_y),
                                font, font_scale, (0, 255, 0), thickness,cv2.LINE_AA)
                    
                    cv2.imshow(WIN_NAME, countdown_frame)
                    cv2.waitKey(1000)

                print("[Game] Start recording Kungfu section")
                recording = True
                recorded_frames = [] 
            elif key in (ord('e'), ord('E')):   
                if recording:
                      print("[Game] Stopped recording. Processing results...")
                      recording = False

                if len(recorded_frames) > 0:   
                               
                    processing_screen = np.zeros((300, 700, 3), dtype=np.uint8)
                    cv2.putText(processing_screen, "Now processing the result...",
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow("Result", processing_screen)
                    cv2.setWindowProperty("Result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.waitKey(1)

                        
                    n, total, grade, feedback = evaluate_session(recorded_frames, references, pose)

                     
                    summary = np.zeros((300, 700, 3), dtype=np.uint8)
                    cv2.putText(summary, f"Matched: {n}/{total}", (40, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, TEXT_COLOR, 3)
                    cv2.putText(summary, f"Grade: {grade}", (40, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, TEXT_COLOR, 3)
                    cv2.putText(summary, feedback, (40, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.imshow("Result", summary)
                    cv2.setWindowProperty("Result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.waitKey(0)
                    cv2.destroyWindow("Result")

                else:   
                    warning_screen = np.zeros((300, 700, 3), dtype=np.uint8)
                    cv2.putText(warning_screen, " No frames recorded!",
                    (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.putText(warning_screen, "Press S to start recording first!",
                    (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("Result", warning_screen)
                    cv2.setWindowProperty("Result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.waitKey(1500)
                    cv2.destroyWindow("Result")


            elif key in (ord('q'), ord('Q')):
                confirm_screen = np.zeros((300, 700, 3), dtype=np.uint8)
                cv2.putText(confirm_screen, "Are you sure you want to quit?",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(confirm_screen, "Press Y to confirm, N to cancel.",
                            (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("Confirm", confirm_screen)
                cv2.setWindowProperty("Confirm", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('y'), ord('Y')):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    elif key in (ord('n'), ord('N')):
                        cv2.destroyWindow("Confirm")
                        break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
