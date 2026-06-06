from collections import deque
import time
from count_fingers import distance

history = deque(maxlen=12)
tap_times = deque(maxlen=6)

tap_phase = "ready"
last_detected_time = 0
last_seen_time = 0


def reset_doremifasolati():
    global tap_phase

    history.clear()
    tap_times.clear()
    tap_phase = "ready"


def hand_metrics(ll):
    palm = distance(ll[0], ll[9]) + 1e-6

    tip_ids = [8, 12, 16, 20]
    base_ids = [5, 9, 13, 17]

    tip_y = sum(ll[i].y for i in tip_ids) / len(tip_ids)
    base_y = sum(ll[i].y for i in base_ids) / len(base_ids)

    center_y = (ll[0].y + ll[5].y + ll[9].y + ll[13].y + ll[17].y) / 5

    # Fingertips moving downward makes this value increase
    relative_y = (tip_y - base_y) / palm

    # Reject closed fist: at least 2 fingers should be somewhat extended/curved
    open_count = 0
    for tip, base in zip(tip_ids, base_ids):
        openness = (distance(ll[tip], ll[0]) - distance(ll[base], ll[0])) / palm
        if openness > 0.10:
            open_count += 1

    return palm, relative_y, center_y, open_count


def is_doremifasolati(hand_res):
    global tap_phase, last_detected_time, last_seen_time

    now = time.time()

    # DOREMIFASOLATI requires exactly two hands
    if not hand_res.multi_hand_landmarks or len(hand_res.multi_hand_landmarks) != 2:
        reset_doremifasolati()
        return False

    hands = hand_res.multi_hand_landmarks

    # Sort by wrist x-position so left/right order stays stable
    hand_landmarks = sorted(
        [hands[0].landmark, hands[1].landmark],
        key=lambda ll: ll[0].x
    )

    ll_left = hand_landmarks[0]
    ll_right = hand_landmarks[1]

    left_palm, left_rel_y, left_center_y, left_open = hand_metrics(ll_left)
    right_palm, right_rel_y, right_center_y, right_open = hand_metrics(ll_right)

    palm_avg = (left_palm + right_palm) / 2

    # Both hands must look like piano-playing hands, not fists
    if left_open < 2 or right_open < 2:
        reset_doremifasolati()
        return False

    # If hands were lost for a while, clear old motion memory
    if now - last_seen_time > 0.6:
        reset_doremifasolati()

    last_seen_time = now

    history.append((
        now,
        left_rel_y,
        right_rel_y,
        left_center_y,
        right_center_y,
        palm_avg
    ))

    # Cooldown after successful detection
    if now - last_detected_time < 0.8:
        return False

    if len(history) < 4:
        return False

    old_t, old_left_rel, old_right_rel, old_left_cy, old_right_cy, old_palm = history[-4]
    current_t, current_left_rel, current_right_rel, current_left_cy, current_right_cy, current_palm = history[-1]

    left_dy = current_left_rel - old_left_rel
    right_dy = current_right_rel - old_right_rel

    combined_old = (old_left_rel + old_right_rel) / 2
    combined_current = (current_left_rel + current_right_rel) / 2
    combined_dy = combined_current - combined_old

    # Reject whole-hand bouncing
    left_hand_move = abs(current_left_cy - old_left_cy) / current_palm
    right_hand_move = abs(current_right_cy - old_right_cy) / current_palm

    if left_hand_move > 0.35 or right_hand_move > 0.35:
        return False

    DOWN_DELTA = 0.055
    UP_DELTA = -0.030

    HAND_DOWN_MIN = 0.015
    HAND_UP_MIN = -0.010

    MIN_TAP_GAP = 0.15
    RHYTHM_WINDOW = 2.2
    REQUIRED_TAPS = 3

    # Both hands move fingers downward: piano key press
    if (
        tap_phase in ["ready", "up"]
        and combined_dy > DOWN_DELTA
        and left_dy > HAND_DOWN_MIN
        and right_dy > HAND_DOWN_MIN
    ):
        tap_phase = "down"

    # Both hands move fingers upward after pressing down: one tap completed
    elif (
        tap_phase == "down"
        and combined_dy < UP_DELTA
        and left_dy < HAND_UP_MIN
        and right_dy < HAND_UP_MIN
    ):
        if not tap_times or now - tap_times[-1] > MIN_TAP_GAP:
            tap_times.append(now)

        tap_phase = "up"

    # Remove old taps
    while tap_times and now - tap_times[0] > RHYTHM_WINDOW:
        tap_times.popleft()

    if len(tap_times) < REQUIRED_TAPS:
        return False

    recent = list(tap_times)[-REQUIRED_TAPS:]
    intervals = []

    for i in range(1, len(recent)):
        intervals.append(recent[i] - recent[i - 1])

    if min(intervals) < MIN_TAP_GAP:
        return False

    if max(intervals) > 1.0:
        return False

    if max(intervals) / max(min(intervals), 1e-6) > 2.2:
        return False

    last_detected_time = now
    reset_doremifasolati()
    return True