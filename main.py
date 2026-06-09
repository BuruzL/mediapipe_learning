import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import json
import os
import random
from collections import deque

# Static gesture modules
from define_handedness import handedness_label
from is_palm_facing import is_palm_facing
from is_bird import is_bird
from is_vowel import is_vowel
from is_drunk import is_drunk
from is_person import is_person
from is_word_ok import is_word_ok
from is_gun import is_gun
from is_song import is_song
from is_anime import is_anime

# Face expression modules
from is_happy import is_happy
from is_sad import is_sad

# Dynamic gesture modules
from is_rhythm import is_rhythm, reset_rhythm
from is_doremifasolati import is_doremifasolati, reset_doremifasolati
from is_movie import is_movie, reset_movie
from is_tvseries import is_tvseries, reset_tvseries


# ====================================================================
# MEDIAPIPE SETUP
# ====================================================================

# Use the MediaPipe Tasks API (hand landmarker model) and provide a small
# compatibility wrapper so the rest of the code — which expects the
# classic `solutions`-style results — continues to work.

# Hand landmarker model (bundled in this workspace)
HAND_MODEL = "models/hand_landmarker.task"
_base_options = python.BaseOptions(model_asset_path=HAND_MODEL)
_hand_options = vision.HandLandmarkerOptions(
    base_options=_base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
)
hand_landmarker = vision.HandLandmarker.create_from_options(_hand_options)

# Face mesh setup: try legacy `solutions` FaceMesh first, then fall back
# to Tasks `FaceLandmarker` if a model is present in `models/`.
FACE_API = 'NONE'
mp_face_mesh = None
mp_draw = None
face_mesh = None

try:
    # Try legacy solutions API (may not be exposed in this packaging)
    mp_face_mesh = mp.solutions.face_mesh
    mp_draw = mp.solutions.drawing_utils
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    FACE_API = 'SOLUTIONS'
except Exception:
    # Try Tasks FaceLandmarker if a model exists
    import os
    FACE_MODEL = os.path.join('models', 'face_landmarker.task')
    if os.path.exists(FACE_MODEL):
        try:
            _face_base = python.BaseOptions(model_asset_path=FACE_MODEL)
            _face_opts = vision.FaceLandmarkerOptions(
                base_options=_face_base,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
            )
            face_landmarker = vision.FaceLandmarker.create_from_options(_face_opts)
            FACE_API = 'TASKS'
        except Exception:
            FACE_API = 'NONE'

if FACE_API == 'NONE':
    class DummyFaceProcessor:
        def process(self, _image):
            class R: pass
            r = R()
            r.multi_face_landmarks = None
            return r

    face_mesh = DummyFaceProcessor()

# Helper: draw hand landmarks using the Tasks-landmarker format
def draw_hand_landmarks(frame, landmarks):
    h, w = frame.shape[:2]

    # Draw points
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # Draw connections
    for conn in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        start = landmarks[conn.start]
        end = landmarks[conn.end]
        x0, y0 = int(start.x * w), int(start.y * h)
        x1, y1 = int(end.x * w), int(end.y * h)
        cv2.line(frame, (x0, y0), (x1, y1), (255, 0, 0), 1)


# ====================================================================
# UNIFIED CHECKERS
# Every checker has this signature:
# (hand_res, face_res, palm_a, palm_b) -> bool
# ====================================================================

def check_bird(hand_res, face_res, pa, pb):
    return is_bird(hand_res, pa, pb)


def check_vowel(hand_res, face_res, pa, pb):
    return is_vowel(hand_res, pa, pb)


def check_drunk(hand_res, face_res, pa, pb):
    return is_drunk(hand_res)


def check_person(hand_res, face_res, pa, pb):
    return is_person(hand_res)


def check_word_ok(hand_res, face_res, pa, pb):
    return is_word_ok(hand_res)


def check_gun(hand_res, face_res, pa, pb):
    return is_gun(hand_res)


# Static: one hand near ear, other hand extended outward
def check_song(hand_res, face_res, pa, pb):
    return is_song(hand_res)


# Static: anime girl V pose under chin
def check_anime(hand_res, face_res, pa, pb):
    return is_anime(hand_res)


# Face expression: smile
def check_happy(hand_res, face_res, pa, pb):
    return is_happy(face_res)


# Face expression: sad/frown lips
def check_sad(hand_res, face_res, pa, pb):
    return is_sad(face_res)


# Dynamic: one-hand piano-like tapping
def check_rhythm(hand_res, face_res, pa, pb):
    return is_rhythm(hand_res)


# Dynamic: two-hand piano-like tapping
def check_doremifasolati(hand_res, face_res, pa, pb):
    return is_doremifasolati(hand_res)


# Dynamic: one hand vertical, other hand fist rolling handle
def check_movie(hand_res, face_res, pa, pb):
    return is_movie(hand_res)


# Dynamic: two index fingers making rectangle
def check_tvseries(hand_res, face_res, pa, pb):
    return is_tvseries(hand_res)


# ====================================================================
# WORD BANK
# ====================================================================

WORD_BANK = {
    "BIRD":           check_bird,
    "VOWEL":          check_vowel,
    "DRUNK":          check_drunk,
    "PERSON":         check_person,
    "OK":             check_word_ok,
    "GUN":            check_gun,
    "SONG":           check_song,
    "ANIME":          check_anime,
    "HAPPY":          check_happy,
    "SAD":            check_sad,
    "RHYTHM":         check_rhythm,
    "DOREMIFASOLATI": check_doremifasolati,
    "MOVIE":          check_movie,
    "TVSERIES":       check_tvseries,
}

# Motion-based signs. These are accepted immediately after their motion finishes.
DYNAMIC_WORDS = {
    "RHYTHM",
    "DOREMIFASOLATI",
    "MOVIE",
    "TVSERIES",
}

# Detection priority matters when more than one checker returns True.
# Put more specific hand signs before broad signs like PERSON or face-only signs.
DETECTION_PRIORITY = [
    # Prefer static PERSON detection before motion-based gestures to
    # avoid mistaking a held-person sign for a rhythmic finger tap.
    "PERSON",
    "RHYTHM",
    "DOREMIFASOLATI",
    "MOVIE",
    "TVSERIES",
    "GUN",
    "SONG",
    "ANIME",
    "BIRD",
    "VOWEL",
    "DRUNK",
    "OK",
    "HAPPY",
    "SAD",
]


# ====================================================================
# MEANING ENGINE
# Add your own rules here.
# ====================================================================

MEANING_RULES = {
    ("GUN", "PERSON"): "GUNMAN",
    ("PERSON", "GUN"): "ARMED PERSON",

    ("RHYTHM", "PERSON"): "DANCER",
    ("DOREMIFASOLATI", "PERSON"): "MUSICIAN",

    ("SONG", "PERSON"): "SINGER",
    ("PERSON", "SONG"): "SINGER",

    ("MOVIE", "PERSON"): "ACTOR",
    ("PERSON", "MOVIE"): "ACTOR",

    ("TVSERIES", "PERSON"): "TV ACTOR",
    ("PERSON", "TVSERIES"): "TV ACTOR",

    ("GUN", "MOVIE"): "ACTION MOVIE",
    ("GUN", "TVSERIES"): "CRIME SERIES",

    ("SAD", "MOVIE"): "SAD MOVIE",
    ("HAPPY", "MOVIE"): "COMEDY MOVIE",

    ("ANIME", "MOVIE"): "ANIME MOVIE",
    ("ANIME", "TVSERIES"): "ANIME SERIES",

    ("DRUNK", "PERSON"): "DRUNK PERSON",
    ("PERSON", "DRUNK"): "DRUNK PERSON",

    ("HAPPY", "PERSON"): "HAPPY PERSON",
    ("PERSON", "HAPPY"): "HAPPY PERSON",

    ("SAD", "PERSON"): "SAD PERSON",
    ("PERSON", "SAD"): "SAD PERSON",

    ("ANIME", "SONG"): "ANIME SONG",
    ("SONG", "ANIME"): "ANIME SONG",

    ("BIRD", "SONG"): "BIRD SONG",
    ("SONG", "BIRD"): "BIRD SONG",

    ("SONG", "RHYTHM"): "MUSIC",
    ("RHYTHM", "DOREMIFASOLATI"): "MELODY",

    ("PERSON", "PERSON"): "PEOPLE",

    ("PERSON", "GUN", "MOVIE"): "ACTION HERO",
    ("PERSON", "SONG", "RHYTHM"): "SINGER / PERFORMER",
    ("PERSON", "DOREMIFASOLATI", "SONG"): "MUSICIAN",
    ("PERSON", "ANIME", "MOVIE"): "ANIME FAN",
    ("PERSON", "HAPPY", "SONG"): "HAPPY SINGER",
    ("PERSON", "SAD", "SONG"): "SAD SINGER",
    ("GUN", "PERSON", "MOVIE"): "GUNMAN IN MOVIE",
    ("MOVIE", "SONG", "RHYTHM"): "MUSICAL MOVIE",
    ("PERSON", "ANIME", "SONG"): "ANIME PERFORMER",
    ("ANIME", "PERSON", "SONG"): "ANIME SINGER",
    ("PERSON", "MOVIE", "SONG"): "MOVIE STAR",
    ("PERSON", "HAPPY", "MOVIE"): "COMEDY ACTOR",
    ("PERSON", "SAD", "MOVIE"): "TRAGEDY ACTOR",
    ("PERSON", "BIRD", "SONG"): "BIRD FAN",

    ("MOVIE", "PERSON", "GUN"): "ACTION STAR",
    ("SONG", "PERSON", "MOVIE"): "MUSIC STAR",
    ("ANIME", "PERSON"): "ANIME FAN",
    ("BIRD", "PERSON"): "BIRD WATCHER",
    ("HAPPY", "SONG"): "HAPPY SONG",
    ("SAD", "SONG"): "SAD SONG",
}


def guess_meaning(sign_stack):
    """Return the best meaning from the latest signs."""
    signs = list(sign_stack)

    # Check last 3 signs first
    if len(signs) >= 3:
        last_three = tuple(signs[-3:])
        if last_three in MEANING_RULES:
            return MEANING_RULES[last_three]

    # Then check last 2 signs
    if len(signs) >= 2:
        last_two = tuple(signs[-2:])
        if last_two in MEANING_RULES:
            return MEANING_RULES[last_two]

    # If there is no combined meaning, return the last word itself
    if len(signs) >= 1:
        return signs[-1]

    return "NO MEANING YET"


# ====================================================================
# CONFIG
# ====================================================================

HOLD_TIME = 0.6          # Static sign must be held for this long
ADD_COOLDOWN = 1.2       # Prevents the same sign from being added every frame
STACK_SIZE = 6
SMOOTHING_LEN = 7
DRAW_FACE_MESH = True

# Dynamic suppression: number of recent frames where any static sign suppresses dynamics
STATIC_SUPPRESSION_FRAMES = 5

# App mode: 'FREE' = free recognition, 'PROMPT' = prompt user with words to act out
MODE = 'FREE'
GAME_DURATION = 60


# ====================================================================
# STATE
# ====================================================================

sign_stack = deque(maxlen=STACK_SIZE)

candidate_word = None
candidate_start = None
last_added_word = None
last_added_time = 0
latest_detected_word = None
palm_history_a = deque(maxlen=SMOOTHING_LEN)
palm_history_b = deque(maxlen=SMOOTHING_LEN)

# Buffer tracking whether a static sign was seen recently
recent_static = deque(maxlen=STATIC_SUPPRESSION_FRAMES)

# Game-mode state
current_word = random.choice(list(WORD_BANK.keys()))
score = 0
hold_start = None
game_start = time.time()
game_over = False
feedback_text = None
feedback_time = 0

# High scores persistence
SCORES_FILE = 'highscores.json'
def load_high_scores():
    if os.path.exists(SCORES_FILE):
        try:
            with open(SCORES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_high_scores(scores):
    try:
        with open(SCORES_FILE, 'w', encoding='utf-8') as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def record_score(score):
    scores = load_high_scores()
    scores.append({'score': score, 'time': int(time.time())})
    scores = sorted(scores, key=lambda s: s['score'], reverse=True)[:10]
    save_high_scores(scores)
    return scores

high_scores = load_high_scores()


# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def stable_majority(history):
    if not history:
        return False
    return sum(history) > len(history) // 2


def reset_dynamic_gestures():
    reset_rhythm()
    reset_doremifasolati()
    reset_movie()
    reset_tvseries()


def can_add_word(word):
    """Avoid duplicate words while the same sign is still visible."""
    now = time.time()

    if now - last_added_time < ADD_COOLDOWN:
        return False

    # Prevent GUN, GUN, GUN while the user is still holding the gun sign.
    if len(sign_stack) > 0 and sign_stack[-1] == word:
        return False

    return True


def add_word_to_stack(word):
    global last_added_word, last_added_time, candidate_word, candidate_start

    if not can_add_word(word):
        return False

    sign_stack.append(word)
    last_added_word = word
    last_added_time = time.time()

    candidate_word = None
    candidate_start = None

    reset_dynamic_gestures()
    return True


def detect_word(hand_res, face_res, pa, pb):
    """Detect any known sign. Returns one word or None."""
    for word in DETECTION_PRIORITY:
        checker = WORD_BANK[word]

        try:
            if checker(hand_res, face_res, pa, pb):
                return word
        except Exception:
            # If one sign checker crashes for a missing landmark,
            # ignore it and continue checking other signs.
            continue

    return None


def choose_next_word(exclude=None):
    choices = [w for w in WORD_BANK.keys() if w != exclude]
    if choices:
        return random.choice(choices)
    return random.choice(list(WORD_BANK.keys()))


def draw_text(frame, text, x, y, scale=0.8, color=(255, 255, 255), thickness=2):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness
    )


def clear_stack():
    global candidate_word, candidate_start, last_added_word, last_added_time, latest_detected_word

    sign_stack.clear()
    candidate_word = None
    candidate_start = None
    last_added_word = None
    last_added_time = 0
    latest_detected_word = None
    reset_dynamic_gestures()


# ====================================================================
# WEBCAM LOOP
# ====================================================================

# Webcam
# Try opening the default camera with multiple backends/indices for robustness.
def open_camera():
    print("Opening camera index 0 with default backend...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("Camera opened with default backend.")
        # Try to set a larger capture resolution for bigger camera space
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        except Exception:
            pass
        return cap

    print("Default backend failed, trying DirectShow...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        print("Camera opened with DirectShow.")
        return cap

    # Try a few other indices in case the device index differs
    for idx in range(1, 4):
        print(f"Trying camera index {idx}...")
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Camera opened at index {idx}.")
            return cap

    return None

cap = open_camera()
if cap is None or not cap.isOpened():
    print("Error: Could not open any camera. Check webcam connection and permissions.")
    # Close the hand landmarker if it was created
    try:
        hand_landmarker.close()
    except Exception:
        pass
    raise SystemExit(1)

# Timestamp for Tasks API video detection
timestamp_ms = 0

# Make the display window resizable and set an initial larger size
WINDOW_NAME = "Sign Meaning Guesser"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
try:
    cv2.resizeWindow(WINDOW_NAME, 1280, 800)
except Exception:
    pass

# Face drawing toggle
DRAW_FACE = True

# Optional camera size
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ------------------------------------------------------------
        # Process hands and face separately (HandLandmarker via Tasks API)
        # ------------------------------------------------------------
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        task_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += 33

        # Wrap Tasks result to match the `solutions`-style interface used
        # by the checker functions elsewhere in the code.
        class HandResultsWrapper:
            pass

        hand_results = HandResultsWrapper()

        if task_result and task_result.hand_landmarks:
            hand_results.multi_hand_landmarks = []
            for landmarks in task_result.hand_landmarks:
                # each `landmarks` is a list of NormalizedLandmark objects
                obj = type("HL", (), {})()
                obj.landmark = landmarks
                hand_results.multi_hand_landmarks.append(obj)

            # Build `multi_handedness` entries mimicking the old API
            hand_results.multi_handedness = []
            for handed in task_result.handedness:
                info = type("Info", (), {})()
                # `handed` is a list of Category objects; pick the first
                class C: pass
                c = C()
                # Category uses `category_name` in Tasks API
                label = None
                if handed and hasattr(handed[0], 'category_name') and handed[0].category_name:
                    label = handed[0].category_name
                elif handed and hasattr(handed[0], 'category_name'):
                    label = handed[0].display_name
                else:
                    label = 'Right'
                c.label = label
                info.classification = [c]
                hand_results.multi_handedness.append(info)
        else:
            hand_results.multi_hand_landmarks = None
            hand_results.multi_handedness = None

        # Face results depending on available API
        if FACE_API == 'SOLUTIONS':
            face_results = face_mesh.process(rgb)
        elif FACE_API == 'TASKS':
            face_task_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            class FaceResultsWrapper:
                pass
            face_results = FaceResultsWrapper()
            if face_task_result and face_task_result.face_landmarks:
                # Wrap into multi_face_landmarks like legacy API
                face_results.multi_face_landmarks = []
                for fl in face_task_result.face_landmarks:
                    obj = type('F', (), {})()
                    obj.landmark = fl
                    face_results.multi_face_landmarks.append(obj)
            else:
                face_results.multi_face_landmarks = None
        else:
            face_results = face_mesh.process(rgb)

        # ------------------------------------------------------------
        # Palm-facing smoothing for two-hand gestures
        # ------------------------------------------------------------
        if (
            hand_results.multi_hand_landmarks
            and len(hand_results.multi_hand_landmarks) == 2
            and hand_results.multi_handedness
        ):
            a, b = hand_results.multi_hand_landmarks
            info_a, info_b = hand_results.multi_handedness

            label_a = handedness_label(info_a)
            label_b = handedness_label(info_b)

            palm_history_a.append(is_palm_facing(a, label_a))
            palm_history_b.append(is_palm_facing(b, label_b))

        else:
            palm_history_a.clear()
            palm_history_b.clear()

        pa = stable_majority(palm_history_a)
        pb = stable_majority(palm_history_b)

        # ------------------------------------------------------------
        # Draw hand skeletons
        # ------------------------------------------------------------
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                draw_hand_landmarks(frame, hand_landmarks.landmark)

        # ------------------------------------------------------------
        # Draw face mesh (if available and enabled)
        # ------------------------------------------------------------
        if DRAW_FACE and DRAW_FACE_MESH and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                if FACE_API == 'SOLUTIONS' and mp_draw is not None:
                    mp_draw.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_draw.DrawingSpec(
                            thickness=1,
                            circle_radius=1
                        )
                    )
                else:
                    # Simple point-draw fallback for Tasks or dummy results
                    h, w = frame.shape[:2]
                    for lm in face_landmarks.landmark:
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        cv2.circle(frame, (x, y), 1, (255, 200, 0), -1)

        # ------------------------------------------------------------
        # Detection + mode handling
        # - FREE mode: user can show any sign; it stacks and builds meanings
        # - PROMPT mode: user is shown a target word to act out and scores
        # ------------------------------------------------------------
        detected_word = detect_word(hand_results, face_results, pa, pb)
        latest_detected_word = detected_word

        now = time.time()

        # Per-frame static-sign check: detect if any static checker is True.
        static_seen = False
        for w, checker in WORD_BANK.items():
            if w in DYNAMIC_WORDS:
                continue
            try:
                if checker(hand_results, face_results, pa, pb):
                    static_seen = True
                    break
            except Exception:
                continue

        recent_static.append(static_seen)

        if MODE == 'FREE':
            if detected_word is None:
                candidate_word = None
                candidate_start = None

            elif detected_word in DYNAMIC_WORDS:
                # Motion-based sign: only accept if no static sign has
                # been seen recently (prevents held static poses from
                # triggering motion detectors).
                if not any(recent_static):
                    add_word_to_stack(detected_word)

            else:
                # Static sign: user must hold it steadily.
                if candidate_word != detected_word:
                    candidate_word = detected_word
                    candidate_start = now
                else:
                    if candidate_start is not None and now - candidate_start >= HOLD_TIME:
                        add_word_to_stack(detected_word)

        elif MODE == 'PROMPT':
            # PROMPT game mode with timer
            elapsed = now - game_start
            remaining = max(0, GAME_DURATION - elapsed)

            if not game_over and elapsed >= GAME_DURATION:
                # End the game
                game_over = True
                high_scores = record_score(score)

            if game_over:
                # During game-over screen we ignore detections and wait for keys
                pass
            else:
                # If the user is not showing the target, reset candidate
                if detected_word is None or detected_word != current_word:
                    candidate_word = None
                    candidate_start = None
                else:
                    # Detected the target sign
                    if detected_word in DYNAMIC_WORDS:
                        if not any(recent_static):
                            accepted = add_word_to_stack(detected_word)
                            if accepted and detected_word == current_word:
                                score += 1
                                feedback_text = "Correct!"
                                feedback_time = now
                                current_word = choose_next_word(current_word)
                    else:
                        if candidate_word != detected_word:
                            candidate_word = detected_word
                            candidate_start = now
                        else:
                            if candidate_start is not None and now - candidate_start >= HOLD_TIME:
                                accepted = add_word_to_stack(detected_word)
                                if accepted and detected_word == current_word:
                                    score += 1
                                    feedback_text = "Correct!"
                                    feedback_time = now
                                    current_word = choose_next_word(current_word)

        # ------------------------------------------------------------
        # UI
        # ------------------------------------------------------------
        h, w = frame.shape[:2]
        meaning = guess_meaning(sign_stack)
        stack_text = " + ".join(sign_stack) if sign_stack else "EMPTY"
        detected_text = latest_detected_word if latest_detected_word else "..."
        added_text = last_added_word if last_added_word else "..."

        # Dark top panel
        cv2.rectangle(frame, (0, 0), (w, 190), (0, 0, 0), -1)

        if MODE == 'FREE':
            draw_text(frame, "FREE SIGN RECOGNITION MODE", 30, 35, 0.9, (0, 255, 255), 2)
            draw_text(frame, f"Detected: {detected_text}", 30, 75, 0.8, (255, 255, 255), 2)
            draw_text(frame, f"Last added: {added_text}", 30, 110, 0.8, (255, 255, 255), 2)
            draw_text(frame, f"Stack: {stack_text}", 30, 145, 0.8, (255, 255, 255), 2)
            draw_text(frame, f"Meaning: {meaning}", 30, 180, 0.8, (0, 255, 0), 2)
        else:
            draw_text(frame, f"PROMPT MODE — Act: {current_word}", 30, 35, 0.9, (0, 200, 255), 2)
            # Show remaining time and score
            elapsed = now - game_start
            remaining = max(0, GAME_DURATION - elapsed)
            draw_text(frame, f"Time: {int(remaining)}s   Score: {score}", 30, 75, 0.8, (255, 255, 255), 2)

            # Color the detected text green when it matches the target
            if detected_text == current_word:
                det_color = (0, 200, 0)
            else:
                det_color = (200, 100, 100)

            draw_text(frame, f"Detected: {detected_text}", 30, 110, 0.8, det_color, 2)
            draw_text(frame, f"Last added: {added_text}", 30, 145, 0.8, (255, 255, 255), 2)

            # Brief feedback overlay when the user scores
            if feedback_text and now - feedback_time < 1.5:
                cv2.rectangle(frame, (w//2 - 220, h//2 - 80), (w//2 + 220, h//2 + 80), (0, 0, 0), -1)
                draw_text(frame, feedback_text, w//2 - 160, h//2, 1.2, (0, 255, 0), 3)

            # If game over, draw full-screen summary
            if game_over:
                cv2.rectangle(frame, (w//4, h//4), (w*3//4, h*3//4), (0, 0, 0), -1)
                draw_text(frame, "GAME OVER", w//2 - 120, h//2 - 120, 1.4, (0, 180, 255), 3)
                draw_text(frame, f"Score: {score}", w//2 - 80, h//2 - 60, 1.0, (255, 255, 255), 2)
                draw_text(frame, "High Scores:", w//2 - 160, h//2 - 20, 0.8, (255, 255, 255), 2)
                # list top 5
                for i, s in enumerate(high_scores[:5]):
                    ts = time.strftime('%Y-%m-%d', time.localtime(s['time']))
                    draw_text(frame, f"{i+1}. {s['score']}  {ts}", w//2 - 120, h//2 + 10 + i*30, 0.7, (200, 200, 200), 2)
                draw_text(frame, "Press R to play again or M to return to FREE mode", w//2 - 240, h//2 + 180, 0.6, (200, 200, 200), 2)

        # Hold-progress bar for static signs
        if candidate_word is not None and candidate_word not in DYNAMIC_WORDS and candidate_start is not None:
            held = now - candidate_start
            progress = min(1.0, held / HOLD_TIME)

            cv2.rectangle(frame, (w - 430, 35), (w - 30, 60), (100, 100, 100), 2)
            cv2.rectangle(frame, (w - 430, 35), (w - 430 + int(400 * progress), 60), (0, 255, 0), -1)
            draw_text(frame, f"Hold: {candidate_word}", w - 430, 90, 0.7, (200, 200, 200), 2)

        draw_text(frame, "C=clear   U=undo   Q=quit", 30, h - 30, 0.7, (180, 180, 180), 2)
        draw_text(frame, "M=toggle mode   N=next word (prompt mode)   F=toggle face draw", 30, h - 65, 0.7, (180, 180, 180), 2)

        # If face mesh not available, show a small warning
        if FACE_API == 'NONE':
            draw_text(frame, "FaceMesh unavailable — add models/face_landmarker.task or enable mediapipe.solutions", 30, h - 95, 0.6, (0, 200, 200), 2)

        cv2.imshow("Sign Meaning Guesser", frame)

        # ------------------------------------------------------------
        # Key handling
        # ------------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('c'):
            clear_stack()

        if key == ord('u'):
            if sign_stack:
                sign_stack.pop()
            candidate_word = None
            candidate_start = None
            reset_dynamic_gestures()
        if key == ord('f'):
            DRAW_FACE = not DRAW_FACE
        if key == ord('m'):
            # Toggle modes
            if MODE == 'FREE':
                MODE = 'PROMPT'
                # reset prompt game state
                current_word = choose_next_word()
                score = 0
                candidate_word = None
                candidate_start = None
                sign_stack.clear()
                game_start = time.time()
                game_over = False
                high_scores = load_high_scores()
            else:
                MODE = 'FREE'
                candidate_word = None
                candidate_start = None
                sign_stack.clear()

        if key == ord('n') and MODE == 'PROMPT':
            current_word = choose_next_word(current_word)
        if key == ord('r') and MODE == 'PROMPT':
            # Restart the prompt game
            score = 0
            sign_stack.clear()
            candidate_word = None
            candidate_start = None
            game_start = time.time()
            game_over = False
            current_word = choose_next_word()

finally:
    cap.release()
    hand_landmarker.close()
    cv2.destroyAllWindows()
