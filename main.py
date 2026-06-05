import cv2
import mediapipe as mp
import random
import time
from collections import deque

# Your detector modules
from count_fingers import count_fingers
from define_handedness import handedness_label
from is_palm_facing import is_palm_facing
from is_bird import is_bird
from is_vowel import is_vowel
from is_drunk import is_drunk
from is_person import is_person
from is_word_ok import is_word_ok


# ====================================================================
# MEDIAPIPE SETUP
# ====================================================================

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)


# ====================================================================
# UNIFIED CHECKERS
# Each checker has the same signature: (hand_res, palm_a, palm_b) -> bool
# This lets us put them all in one dictionary.
# ====================================================================

def check_bird(hand_res, pa, pb):
    return is_bird(hand_res, pa, pb)

def check_vowel(hand_res, pa, pb):
    return is_vowel(hand_res, pa, pb)

def check_drunk(hand_res, pa, pb):
    return is_drunk(hand_res)              # ignores palm flags

def check_person(hand_res, pa, pb):
    return is_person(hand_res)

def check_word_ok(hand_res, pa, pb):
    return is_word_ok(hand_res)

# Word bank: just add/remove entries to change the game
WORD_BANK = {
    "BIRD":    check_bird,
    "VOWEL":   check_vowel,
    "DRUNK":   check_drunk,
    "PERSON":  check_person,
    "OK":      check_word_ok,
}


# ====================================================================
# GAME CONFIG
# ====================================================================

HOLD_TIME      = 0.5     # seconds to hold a pose before it counts
GAME_DURATION  = 60      # total game length in seconds
SMOOTHING_LEN  = 7       # frames of palm-facing smoothing


# ====================================================================
# GAME STATE
# ====================================================================

current_word = random.choice(list(WORD_BANK.keys()))
score        = 0
hold_start   = None
game_start   = time.time()
game_over    = False

palm_history_a = deque(maxlen=SMOOTHING_LEN)
palm_history_b = deque(maxlen=SMOOTHING_LEN)


def stable_majority(history):
    if not history:
        return False
    return sum(history) > len(history) // 2


def new_word(exclude=None):
    """Pick a random word, optionally not the one we just had."""
    choices = [w for w in WORD_BANK.keys() if w != exclude]
    return random.choice(choices) if choices else random.choice(list(WORD_BANK.keys()))


# ====================================================================
# WEBCAM LOOP
# ====================================================================

cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # --- Always update palm-facing history when 2 hands are visible ---
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2 and results.multi_handedness:
        a, b = results.multi_hand_landmarks
        info_a, info_b = results.multi_handedness
        label_a = handedness_label(info_a)
        label_b = handedness_label(info_b)
        palm_history_a.append(is_palm_facing(a, label_a))
        palm_history_b.append(is_palm_facing(b, label_b))
    else:
        palm_history_a.clear()
        palm_history_b.clear()

    pa = stable_majority(palm_history_a)
    pb = stable_majority(palm_history_b)

    # --- Draw hand skeletons (helpful visual feedback) ---
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # --- Time tracking ---
    elapsed   = time.time() - game_start
    remaining = max(0, GAME_DURATION - elapsed)
    if remaining == 0:
        game_over = True

    # --- Gesture matching + hold timer ---
    if not game_over:
        checker = WORD_BANK[current_word]
        matched = checker(results, pa, pb)

        if matched:
            if hold_start is None:
                hold_start = time.time()
            elif time.time() - hold_start >= HOLD_TIME:
                score += 1
                hold_start = None
                current_word = new_word(exclude=current_word)
        else:
            hold_start = None

    # --- UI ---
    h, w = frame.shape[:2]

    if game_over:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "GAME OVER", (w // 2 - 240, h // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
        cv2.putText(frame, f"Final Score: {score}", (w // 2 - 200, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        cv2.putText(frame, "Press R to restart, Q to quit", (w // 2 - 250, h // 2 + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    else:
        cv2.putText(frame, f"Act out: {current_word}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(frame, f"Score: {score}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {int(remaining)}s", (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "S=skip   Q=quit", (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        # Hold-progress bar
        if hold_start is not None:
            held = time.time() - hold_start
            progress = min(1.0, held / HOLD_TIME)
            cv2.rectangle(frame, (30, 170), (430, 195), (100, 100, 100), 2)
            cv2.rectangle(frame, (30, 170), (30 + int(400 * progress), 195),
                          (0, 255, 0), -1)

    cv2.imshow("Charades", frame)

    # --- Key handling ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s') and not game_over:
        current_word = new_word(exclude=current_word)
        hold_start = None
    if key == ord('r') and game_over:
        # Restart the game
        score = 0
        hold_start = None
        game_start = time.time()
        game_over = False
        current_word = random.choice(list(WORD_BANK.keys()))
        palm_history_a.clear()
        palm_history_b.clear()

cap.release()
cv2.destroyAllWindows()