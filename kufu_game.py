
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
        feedback = "You failed press Exit to try again"
    elif grade == "D":
        feedback = "You pass press Exit to continue"
    elif grade == "C":
        feedback = "Well done, you press Exit to continue"
    elif grade == "B":
        feedback = "Excellent, great job press Exit to continue"
    elif grade == "A":
        feedback = "Perfect, matched all steps! press Exit to continue"
    else:
        feedback = "Almost pass press Exit to continue"
    print(f"[Game] Matched {n_matched}/{total_refs} reference poses")
    print(f"[Game] Final Grade: {grade}")
    print("[Game] " + feedback)
    return n_matched, total_refs, grade, feedback


def confirm_quit(cap):
    """Show a fullscreen confirmation dialog. Returns True if user confirms quit."""
    h, w = 300, 700
    confirm_screen = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(confirm_screen, "Are you sure you want to quit?",
                (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(confirm_screen, "Click Yes or No, or press Y/N.",
                (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    
    btn_w, btn_h = 140, 60
    yes_x1, yes_y1 = 120, 200
    yes_x2, yes_y2 = yes_x1 + btn_w, yes_y1 + btn_h
    no_x1, no_y1 = 440, 200
    no_x2, no_y2 = no_x1 + btn_w, no_y1 + btn_h

    cv2.rectangle(confirm_screen, (yes_x1, yes_y1), (yes_x2, yes_y2), (0, 200, 0), -1)
    cv2.putText(confirm_screen, "Yes", (yes_x1 + 35, yes_y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.rectangle(confirm_screen, (no_x1, no_y1), (no_x2, no_y2), (0, 0, 200), -1)
    cv2.putText(confirm_screen, "No", (no_x1 + 45, no_y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Confirm", confirm_screen)
    try:
        cv2.setWindowProperty("Confirm", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        
        pass

    
    result = {"confirmed": None}

    def _mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if yes_x1 <= x <= yes_x2 and yes_y1 <= y <= yes_y2:
                result["confirmed"] = True
            elif no_x1 <= x <= no_x2 and no_y1 <= y <= no_y2:
                result["confirmed"] = False

    cv2.setMouseCallback("Confirm", _mouse)

    while True:
        k = cv2.waitKey(1) & 0xFF
        
        if k in (ord('y'), ord('Y')):
            result["confirmed"] = True
        elif k in (ord('n'), ord('N')):
            result["confirmed"] = False

        if result["confirmed"] is not None:
            if result["confirmed"]:
                try:
                    cap.release()
                except Exception:
                    pass
                cv2.destroyAllWindows()
                return True
            else:
                cv2.destroyWindow("Confirm")
                return False



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
    clicked_button = [None]  

    
    button_start = [0, 0, 0, 0]
    button_stop  = [0, 0, 0, 0]
    button_quit  = [0, 0, 0, 0]

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if button_start[0] <= x <= button_start[2] and button_start[1] <= y <= button_start[3]:
                clicked_button[0] = "start"
            elif button_stop[0] <= x <= button_stop[2] and button_stop[1] <= y <= button_stop[3]:
                clicked_button[0] = "stop"
            elif button_quit[0] <= x <= button_quit[2] and button_quit[1] <= y <= button_quit[3]:
                clicked_button[0] = "quit"

    cv2.setMouseCallback(WIN_NAME, mouse_callback)

    with mp_pose.Pose(
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        print("Click Start / Stop / Quit on screen")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Error] Failed to read camera")
                break

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            h, w = frame.shape[:2]

            
            btn_height = 50
            margin = 20
            btn_y1 = h - btn_height - margin
            btn_y2 = h - margin
            btn_width = (w - 4 * margin) // 3

            button_start[:] = [margin, btn_y1, margin + btn_width, btn_y2]
            button_stop[:]  = [2*margin + btn_width, btn_y1, 2*margin + 2*btn_width, btn_y2]
            button_quit[:]  = [3*margin + 2*btn_width, btn_y1, 3*margin + 3*btn_width, btn_y2]

            
            cv2.rectangle(frame, tuple(button_start[:2]), tuple(button_start[2:]), (0,255,0), -1)
            cv2.putText(frame, "Start", (button_start[0]+20, button_start[1]+35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

            cv2.rectangle(frame, tuple(button_stop[:2]), tuple(button_stop[2:]), (0,255,255), -1)
            cv2.putText(frame, "Stop & Result", (button_stop[0]+10, button_stop[1]+35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

            cv2.rectangle(frame, tuple(button_quit[:2]), tuple(button_quit[2:]), (0,0,255), -1)
            cv2.putText(frame, "Quit", (button_quit[0]+30, button_quit[1]+35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            if recording:
                cv2.putText(frame, "REC", (w - 100, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
                recorded_frames.append(frame.copy())

            cv2.imshow(WIN_NAME, frame)

           
            if clicked_button[0] == "start":
                print("[Game] Starting after 5 second countdown...")
                for i in range(5,0,-1):
                    countdown_frame = np.zeros((h, w, 3), dtype=np.uint8)
                    text = f"{i}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 5
                    thickness = 8
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    text_x = (w - text_size[0]) // 2
                    text_y = (h + text_size[1]) // 2
                    cv2.putText(countdown_frame, text, (text_x, text_y),
                                font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                    cv2.imshow(WIN_NAME, countdown_frame)
                    cv2.waitKey(1000)
                print("[Game] Start recording Kungfu section")
                recording = True
                recorded_frames = []
                clicked_button[0] = None

            elif clicked_button[0] == "stop":
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
                   
                    btn_w, btn_h = 160, 60
                    margin = 20
                    
                    ex_x2 = 700 - margin
                    ex_y1 = margin
                    ex_x1 = ex_x2 - btn_w
                    ex_y2 = ex_y1 + btn_h
                    cv2.rectangle(summary, (ex_x1, ex_y1), (ex_x2, ex_y2), (50, 50, 50), -1)
                    text_size = cv2.getTextSize("Exit", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    tx = ex_x1 + (btn_w - text_size[0]) // 2
                    ty = ex_y1 + (btn_h + text_size[1]) // 2
                    cv2.putText(summary, "Exit", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow("Result", summary)
                    try:
                        cv2.setWindowProperty("Result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    except Exception:
                        pass

                    exit_state = {"exit": False}

                    def _result_mouse(event, x, y, flags, param):
                        if event == cv2.EVENT_LBUTTONDOWN:
                            if ex_x1 <= x <= ex_x2 and ex_y1 <= y <= ex_y2:
                                exit_state["exit"] = True

                    cv2.setMouseCallback("Result", _result_mouse)

                    
                    while True:
                        k = cv2.waitKey(1) & 0xFF
                        if exit_state["exit"]:
                            break
                        if k == 27:  
                            break

                    cv2.destroyWindow("Result")
                else:
                    warning_screen = np.zeros((300, 700, 3), dtype=np.uint8)
                    cv2.putText(warning_screen, " No frames recorded!",
                                (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.putText(warning_screen, "Press Start to record first!",
                                (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("Result", warning_screen)
                    cv2.setWindowProperty("Result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.waitKey(1500)
                    cv2.destroyWindow("Result")
                clicked_button[0] = None

            elif clicked_button[0] == "quit":
                
                if confirm_quit(cap):
                    return
                clicked_button[0] = None
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key in (ord('q'), ord('Q')):
                if confirm_quit(cap):
                    return

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()




