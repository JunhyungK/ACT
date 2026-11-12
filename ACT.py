import time, math, collections
import datetime as dt

import cv2, numpy as np, pandas as pd, mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

from event_predictor import EventPredictor

# ---------- what the system knows about (COCO class names) ----------

PORTABLE = {"backpack", "handbag", "suitcase", "bottle", "cup",
            "cell phone", "book", "pen", "pencil", "laptop",
            "keyboard", "mouse", "remote",
            "toothbrush", "bowl", "fork",
            "knife", "spoon"}

SEATS = {"chair", "couch", "bed"}

FACE_USE = {"cup": "drinking from",
            "bottle": "drinking from",
            "cell phone": "using",
            "fork": "using",
            "knife": "using",
            "spoon": "using",
            "toothbrush": "using",
            "laptop": "using",
            "keyboard": "using",
            "mouse": "using"}

SUGGESTIONS = {
    "picking up":    "Log pickup event",
    "holding":       "Object in use / being carried",
    "putting down":  "Log object released",
    "reaching for":  "Person about to interact",
    "touching":      "Contact with object",
    "drinking from": "Hydration detected",
    "using":         "Device in use",
    "sitting":       "Person seated",
    "":              "",
}

# ---------- tunable settings ----------
YOLO_CONF       = 0.3
NEAR_FRAC       = 0.35   # "near" = wrist within 35% of box's larger side
RISE_FRAC       = 0.04   # object must rise 4% of frame height while touched
RELEASE_GAP     = 5      # frames without a nearby wrist before "released"
PICK_SHOW       = 15     # frames to keep showing "picking up"
PUT_SHOW        = 15     # frames to keep showing "putting down"
REACH_FRAMES    = 3      # near for fewer frames than this = "reaching"
FACE_FRAC       = 0.20   # object within 20% of frame height of mouth
SIT_KNEE_MARGIN = 0.10
SMOOTH          = 3
WRISTS, HIPS, KNEES, MOUTH = (15, 16), (23, 24), (25, 26), (9, 10)

POSE_CONNECTION = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


# ---------- per-object state ----------

class ObjState:
    def __init__(self):
        self.base = None
        self.held = False
        self.gap = 0
        self.near_count = 0
        self.pick_until = -1
        self.put_until = -1
        self.last_seen = 0


def point_box_dist(px, py, box):
    x1, y1, x2, y2 = box
    return math.hypot(max(x1 - px, 0, px - x2), max(y1 - py, 0, py - y2))


def vis_ok(lm, t=0.4):
    return getattr(lm, "visibility", 1.0) > t


def make_landmarker():
    return vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path="POSES/pose_landmarker_heavy.task"),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


# input_options: 0 = video file, 1 = webcam
# camera_index : index of the webcam to use (default 0)
# predictor    : an EventPredictor, or None to run the vision code only
def main(video_path, save_path=None, mirror=False, display_width=960,
         input_options=None, camera_index=0, predictor=None):
    pose_landmarker = make_landmarker()
    yolo = YOLO("YOLO/yolov8s.pt")
    CLASSES = yolo.names
    if isinstance(CLASSES, list):
        CLASSES = dict(enumerate(CLASSES))

    OBJECT_CLASS_IDS = [cid for cid, cname in CLASSES.items() if cname is not None]

    if input_options == 0:
        if video_path is None:
            pose_landmarker.close()
            raise ValueError("video_path is required when input_options=0")
        cap = cv2.VideoCapture(video_path)
        source_name = video_path
    elif input_options == 1:
        cap = cv2.VideoCapture(camera_index)
        source_name = f"Webcam {camera_index}"
    else:
        pose_landmarker.close()
        raise ValueError("input_options must be 0 (video) or 1 (webcam)")

    if not cap.isOpened():
        cap.release()
        pose_landmarker.close()
        raise RuntimeError(f"Cannot open input: {source_name}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or input_options == 1:
        fps = 30.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if input_options == 0 else 0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"{source_name}: {W}x{H}, {fps:.1f} fps, "
          f"{total if input_options == 0 else 'live'} frames")

    writer = None
    if save_path:
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    states = {}
    votes = collections.deque(maxlen=SMOOTH)
    last_ts = -1
    frame_index = 0

    records = []           # one dict per frame
    events = []            # one dict per action segment (label changes)
    cur = None             # current segment
    last_pred_text = ""    # prediction overlay that stays on screen
    pred_every = max(1, int(round(fps)))   # run the model about once per second

    def close_segment(seg, end_s):
        """Finish a segment, run the model on it, and store it as an event."""
        dwell = round(end_s - seg["start"], 2)
        ev = dict(label=seg["label"], start_s=seg["start"], end_s=end_s, dur_s=dwell)
        if predictor and seg["verb"] not in predictor.skip:
            pred, conf, _ = predictor.predict(seg["verb"], seg["obj"], dwell, seg["start"])
            ev.update(pred_event=pred, pred_conf=round(conf, 3))
        if predictor:
            predictor.close_segment(seg["verb"])
        events.append(ev)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break                                   # end of video
            if mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_index += 1

            # timestamp comes from the frame number for video, wall clock for webcam
            if input_options == 1:
                timestamp_ms = max(int(time.monotonic() * 1000), last_ts + 1)
            else:
                timestamp_ms = max(int((frame_index - 1) * 1000 / fps), last_ts + 1)
            last_ts = timestamp_ms

            res = pose_landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)

            # ---- 1. YOLO detection + tracking ----
            tracking_results = yolo.track(frame, persist=True, conf=YOLO_CONF,
                                          classes=OBJECT_CLASS_IDS, verbose=False)[0]
            objs = []                                   # (id, name, box)

            if (tracking_results.boxes is not None
                    and tracking_results.boxes.id is not None):
                for detection_box, track_id in zip(
                        tracking_results.boxes,
                        tracking_results.boxes.id.int().tolist()):
                    class_id = int(detection_box.cls[0])
                    objs.append((track_id, CLASSES[class_id],
                                 detection_box.xyxy[0].tolist()))

            # ---- 2. Pose ----
            wrists, mouth, hip_pt, knee_y = [], None, None, None
            if res.pose_landmarks:
                pose = res.pose_landmarks[0]            # single person
                for a, b in POSE_CONNECTION:
                    cv2.line(frame, (int(pose[a].x * w), int(pose[a].y * h)),
                                    (int(pose[b].x * w), int(pose[b].y * h)), (255, 255, 0), 2)
                for lm in pose:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)

                wrists = [(pose[i].x * w, pose[i].y * h) for i in WRISTS if vis_ok(pose[i])]
                mouth = (np.mean([pose[i].x for i in MOUTH]) * w,
                         np.mean([pose[i].y for i in MOUTH]) * h)
                if all(vis_ok(pose[i]) for i in HIPS):
                    hip_pt = (np.mean([pose[i].x for i in HIPS]) * w,
                              np.mean([pose[i].y for i in HIPS]) * h)
                if all(vis_ok(pose[i]) for i in KNEES):
                    knee_y = np.mean([pose[i].y for i in KNEES]) * h

            # ---- 3. Sitting ----
            sitting = False
            if hip_pt:
                for _, name, (x1, y1, x2, y2) in objs:
                    if name in SEATS and x1 < hip_pt[0] < x2 and y1 < hip_pt[1] < y2:
                        if knee_y is None or abs(knee_y - hip_pt[1]) < SIT_KNEE_MARGIN * h:
                            sitting = True

            # ---- 4. Per-object interaction state machine ----
            candidates = []                             # (priority, verb, text)
            for tracking_id, name, box in objs:
                x1, y1, x2, y2 = box
                color = (0, 165, 255) if name in PORTABLE else (200, 200, 200)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{name} #{tracking_id}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if name not in PORTABLE:
                    continue

                object_state = states.setdefault(tracking_id, ObjState())
                object_state.last_seen = frame_index
                obj_center_y = (y1 + y2) / 2

                near_px = NEAR_FRAC * max(x2 - x1, y2 - y1)
                near = bool(wrists) and min(point_box_dist(x, y, box) for x, y in wrists) < near_px

                if near:
                    object_state.gap = 0
                    object_state.near_count += 1
                    if object_state.base is None:
                        object_state.base = obj_center_y
                    if not object_state.held and (object_state.base - obj_center_y) > RISE_FRAC * h:
                        object_state.held = True
                        object_state.pick_until = frame_index + PICK_SHOW
                else:
                    object_state.gap += 1
                    object_state.near_count = 0
                    if object_state.gap > RELEASE_GAP:
                        if object_state.held:
                            object_state.put_until = frame_index + PUT_SHOW
                        object_state.held = False
                        object_state.base = None

                if object_state.held and frame_index < object_state.pick_until:
                    candidates.append((6, "picking up", f"picking up {name}"))

                elif object_state.held:
                    near_face = mouth and point_box_dist(*mouth, box) < FACE_FRAC * h
                    if near_face and name in FACE_USE:
                        v = FACE_USE[name]
                        candidates.append((4, v, f"{v} {name}"))
                    else:
                        candidates.append((3, "holding", f"holding {name}"))

                elif frame_index < object_state.put_until:
                    candidates.append((5, "putting down", f"putting down {name}"))

                elif near and name == "laptop" and sitting:
                    candidates.append((4, "using", "using laptop"))

                elif near:
                    v = "reaching for" if object_state.near_count < REACH_FRAMES else "touching"
                    candidates.append((2, v, f"{v} {name}"))

            if sitting:
                candidates.append((1, "sitting", "sitting"))

            for t in [t for t, s in states.items() if frame_index - s.last_seen > 60]:
                del states[t]                           # forget objects gone ~2 s

            # ---- 5. Smooth + record ----
            best = max(candidates, key=lambda c: c[0]) if candidates else (0, "", "none")
            votes.append(best[1:])
            (verb, text), _ = collections.Counter(votes).most_common(1)[0]

            t_now = round((frame_index - 1) / fps, 3)

            # object involved in the winning action (handles "cell phone" etc.)
            obj_name = max((n for _, n, _ in objs if text.endswith(n)),
                           key=len, default=None)

            records.append(dict(
                frame=frame_index,
                time_s=t_now,
                action=verb or "none",
                label=text,
                object=obj_name,
                raw_label=best[2],
                objects_visible=[(tid, n) for tid, n, _ in objs],
                n_wrists=len(wrists),
                sitting=sitting,
            ))

            # ---- 6. Segment tracking + model prediction ----
            if cur is None or cur["label"] != text:
                if cur is not None:
                    close_segment(cur, t_now)
                cur = dict(label=text, verb=verb or "none", obj=obj_name, start=t_now)

            # running prediction on the in-progress segment, about once per second
            if (predictor and cur["verb"] not in predictor.skip
                    and frame_index % pred_every == 0):
                pred, conf, _ = predictor.predict(cur["verb"], cur["obj"],
                                                  t_now - cur["start"], t_now)
                last_pred_text = f"Predicted event: {pred} ({conf:.0%})"

            # ---- 7. Overlay + display ----
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.putText(frame, f"Action: {text}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, SUGGESTIONS.get(verb, ""), (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, f"frame {frame_index}/{total}", (w - 170, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            if last_pred_text:
                cv2.putText(frame, last_pred_text, (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if writer:
                writer.write(frame)

            disp = frame
            if display_width and w > display_width:
                disp = cv2.resize(frame, (display_width, int(h * display_width / w)))
            cv2.imshow("Action detector", disp)
            if cv2.waitKey(1) & 0xFF == 27:             # Esc to quit early
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        pose_landmarker.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    if cur is not None:
        close_segment(cur, round((frame_index - 1) / fps, 3))
    return records, events


if __name__ == "__main__":
    # ---------------- configuration ----------------
    MODEL_PATH = "artifacts/event_model.joblib"

    # Context a camera can't see. Use values that exist in dataset.csv.
    CONTEXT = dict(
        age=23,
        occupation="student_grad",
        zone="bed",
        is_school_holiday=0,
        is_off_routine=0,
    )

    # RECORDED VIDEO (testing): pretend the video started at this time.
    # LIVE WEBCAM: set START_TIME = None and INPUT_OPTIONS = 1 (uses the real clock).
    START_TIME = dt.datetime(2026, 11, 10, 18, 30)
    INPUT_OPTIONS = 0            # 0 = video file, 1 = webcam
    VIDEO_PATH = "resources/sitting.mov"

    predictor = EventPredictor(MODEL_PATH, CONTEXT, start_time=START_TIME)

    records, events = main(
        VIDEO_PATH,
        save_path="output_annotated.mp4",
        mirror=True,
        display_width=960,
        input_options=INPUT_OPTIONS,
        camera_index=0,
        predictor=predictor,
    )

    df = pd.DataFrame(records)
    ev = pd.DataFrame(events)

    print(ev)                                    # timeline of actions + predicted events
    print(df.action.value_counts())              # how often each action occurred
    print(df[df.action == "picking up"])         # every frame of a specific action
    df.to_csv("frames.csv", index=False)
    ev.to_csv("events.csv", index=False)