import cv2
import mediapipe as mp
import random
import time
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

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)


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

# Dynamic gestures are motion-based, not hold-based.
# HAPPY and SAD are static face expressions, so do NOT add them here.
DYNAMIC_WORDS = {
    "RHYTHM",
    "DOREMIFASOLATI",
    "MOVIE",
    "TVSERIES",
}


# ====================================================================
# GAME CONFIG
# ====================================================================

HOLD_TIME = 0.5
GAME_DURATION = 60
SMOOTHING_LEN = 7

DRAW_FACE_MESH = True


# ====================================================================
# GAME STATE
# ====================================================================

current_word = random.choice(list(WORD_BANK.keys()))
score = 0
hold_start = None
game_start = time.time()
game_over = False

palm_history_a = deque(maxlen=SMOOTHING_LEN)
palm_history_b = deque(maxlen=SMOOTHING_LEN)


def stable_majority(history):
    if not history:
        return False

    return sum(history) > len(history) // 2


def new_word(exclude=None):
    choices = [w for w in WORD_BANK.keys() if w != exclude]

    if choices:
        return random.choice(choices)

    return random.choice(list(WORD_BANK.keys()))


def reset_dynamic_gestures():
    reset_rhythm()
    reset_doremifasolati()
    reset_movie()
    reset_tvseries()


# ====================================================================
# WEBCAM LOOP
# ====================================================================

cap = cv2.VideoCapture(0)

# Optional camera size
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------
    # Process hands and face separately
    # ------------------------------------------------------------
    hand_results = hands.process(rgb)
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
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # ------------------------------------------------------------
    # Draw face mesh
    # ------------------------------------------------------------
    if DRAW_FACE_MESH and face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
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

    # ------------------------------------------------------------
    # Time tracking
    # ------------------------------------------------------------
    elapsed = time.time() - game_start
    remaining = max(0, GAME_DURATION - elapsed)

    if remaining == 0:
        game_over = True

    # ------------------------------------------------------------
    # Gesture / expression matching
    # ------------------------------------------------------------
    if not game_over:
        checker = WORD_BANK[current_word]
        matched = checker(hand_results, face_results, pa, pb)

        # Dynamic gestures: score immediately when motion is completed
        if current_word in DYNAMIC_WORDS:
            if matched:
                score += 1
                hold_start = None
                current_word = new_word(exclude=current_word)
                reset_dynamic_gestures()

        # Static gestures / face expressions: use hold timer
        else:
            if matched:
                if hold_start is None:
                    hold_start = time.time()

                elif time.time() - hold_start >= HOLD_TIME:
                    score += 1
                    hold_start = None
                    current_word = new_word(exclude=current_word)
                    reset_dynamic_gestures()

            else:
                hold_start = None

    # ------------------------------------------------------------
    # UI
    # ------------------------------------------------------------
    h, w = frame.shape[:2]

    if game_over:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 0), -1)

        cv2.putText(
            frame,
            "GAME OVER",
            (w // 2 - 240, h // 2 - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 0, 255),
            6
        )

        cv2.putText(
            frame,
            f"Final Score: {score}",
            (w // 2 - 200, h // 2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            4
        )

        cv2.putText(
            frame,
            "Press R to restart, Q to quit",
            (w // 2 - 250, h // 2 + 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2
        )

    else:
        cv2.putText(
            frame,
            f"Act out: {current_word}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3
        )

        cv2.putText(
            frame,
            f"Score: {score}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Time: {int(remaining)}s",
            (30, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "S=skip   Q=quit",
            (30, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (180, 180, 180),
            2
        )

        cv2.putText(
            frame,
            "Hands + FaceMesh active",
            (30, h - 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (180, 180, 180),
            2
        )

        # Hold-progress bar only for static gestures / expressions
        if hold_start is not None and current_word not in DYNAMIC_WORDS:
            held = time.time() - hold_start
            progress = min(1.0, held / HOLD_TIME)

            cv2.rectangle(
                frame,
                (30, 170),
                (430, 195),
                (100, 100, 100),
                2
            )

            cv2.rectangle(
                frame,
                (30, 170),
                (30 + int(400 * progress), 195),
                (0, 255, 0),
                -1
            )

    cv2.imshow("Charades", frame)

    # ------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('s') and not game_over:
        current_word = new_word(exclude=current_word)
        hold_start = None
        reset_dynamic_gestures()

    if key == ord('r') and game_over:
        score = 0
        hold_start = None
        game_start = time.time()
        game_over = False
        current_word = random.choice(list(WORD_BANK.keys()))

        palm_history_a.clear()
        palm_history_b.clear()
        reset_dynamic_gestures()


cap.release()
hands.close()
face_mesh.close()
cv2.destroyAllWindows()