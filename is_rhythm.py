from collections import deque
import time
from count_fingers import distance

history = deque(maxlen=12)
tap_times = deque(maxlen=6)

tap_phase = "ready"
last_detected_time = 0
last_seen_time = 0


def reset_rhythm():
    global tap_phase

    history.clear()
    tap_times.clear()
    tap_phase = "ready"


def is_rhythm(hand_res):
    global tap_phase, last_detected_time, last_seen_time

    now = time.time()

    # RHYTHM uses exactly one hand
    if not hand_res.multi_hand_landmarks or len(hand_res.multi_hand_landmarks) != 1:
        reset_rhythm()
        return False

    ll = hand_res.multi_hand_landmarks[0].landmark

    palm = distance(ll[0], ll[9]) + 1e-6

    # If hand was lost, reset old motion memory
    if now - last_seen_time > 0.6:
        reset_rhythm()

    last_seen_time = now

    # Fingertips: index, middle, ring, pinky
    tip_ids = [8, 12, 16, 20]

    # Finger base joints
    base_ids = [5, 9, 13, 17]

    tip_y = sum(ll[i].y for i in tip_ids) / len(tip_ids)
    base_y = sum(ll[i].y for i in base_ids) / len(base_ids)

    # Hand center, used to reject whole-hand movement
    center_x = (ll[0].x + ll[5].x + ll[9].x + ll[13].x + ll[17].x) / 5
    center_y = (ll[0].y + ll[5].y + ll[9].y + ll[13].y + ll[17].y) / 5

    # Relative finger position
    # MediaPipe y increases downward.
    # Fingertips moving down means this value increases.
    relative_y = (tip_y - base_y) / palm

    history.append((now, relative_y, center_x, center_y))

    # Cooldown after successful rhythm detection
    if now - last_detected_time < 0.8:
        return False

    if len(history) < 4:
        return False

    old_t, old_relative_y, old_cx, old_cy = history[-4]
    current_t, current_relative_y, current_cx, current_cy = history[-1]

    finger_dy = current_relative_y - old_relative_y

    # Reject if the whole hand is moving too much.
    # Piano/rhythm should mainly be finger motion, not full hand waving.
    whole_hand_move = abs(current_cy - old_cy) / palm

    if whole_hand_move > 0.35:
        return False

    DOWN_DELTA = 0.06
    UP_DELTA = -0.035

    MIN_TAP_GAP = 0.15
    RHYTHM_WINDOW = 2.0
    REQUIRED_TAPS = 3

    # Finger moves downward: key press
    if tap_phase in ["ready", "up"] and finger_dy > DOWN_DELTA:
        tap_phase = "down"

    # Finger moves upward after down movement: one tap completed
    elif tap_phase == "down" and finger_dy < UP_DELTA:
        if not tap_times or now - tap_times[-1] > MIN_TAP_GAP:
            tap_times.append(now)

        tap_phase = "up"

    # Remove old taps
    while tap_times and now - tap_times[0] > RHYTHM_WINDOW:
        tap_times.popleft()

    if len(tap_times) < REQUIRED_TAPS:
        return False

    # Check rhythm consistency
    recent = list(tap_times)[-REQUIRED_TAPS:]
    intervals = []

    for i in range(1, len(recent)):
        intervals.append(recent[i] - recent[i - 1])

    if min(intervals) < MIN_TAP_GAP:
        return False

    if max(intervals) > 0.9:
        return False

    if max(intervals) / max(min(intervals), 1e-6) > 2.0:
        return False

    last_detected_time = now
    reset_rhythm()
    return True