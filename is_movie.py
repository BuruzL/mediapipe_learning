from collections import deque
import time
import math
from count_fingers import distance

history = deque(maxlen=30)

last_detected_time = 0
last_seen_time = 0


def reset_movie():
    history.clear()


def palm_size(ll):
    return distance(ll[0], ll[9]) + 1e-6


def hand_center(ll):
    x = (ll[0].x + ll[5].x + ll[9].x + ll[13].x + ll[17].x) / 5
    y = (ll[0].y + ll[5].y + ll[9].y + ll[13].y + ll[17].y) / 5
    return x, y


def angle_diff(a, b):
    diff = b - a

    while diff > math.pi:
        diff -= 2 * math.pi

    while diff < -math.pi:
        diff += 2 * math.pi

    return diff


def open_hand_score(ll):
    """
    Higher score means the hand looks open.
    Used for the vertical/sideways static hand.
    """

    palm = palm_size(ll)

    tip_ids = [8, 12, 16, 20]
    base_ids = [5, 9, 13, 17]

    score = 0

    for tip, base in zip(tip_ids, base_ids):
        openness = distance(ll[tip], ll[base]) / palm

        if openness > 0.35:
            score += 1

    return score


def fist_score(ll):
    """
    Higher score means the hand looks like a fist.
    """

    palm = palm_size(ll)

    tip_ids = [8, 12, 16, 20]
    base_ids = [5, 9, 13, 17]

    score = 0

    for tip, base in zip(tip_ids, base_ids):
        tip_dist = distance(ll[tip], ll[0])
        base_dist = distance(ll[base], ll[0])

        curl = (tip_dist - base_dist) / palm
        tip_to_base = distance(ll[tip], ll[base]) / palm

        # Relaxed fist condition
        if curl < 0.35 or tip_to_base < 0.55:
            score += 1

    return score


def choose_static_and_fist_hand(ll0, ll1):
    """
    Decide which hand is the static open hand
    and which hand is the rotating fist.
    """

    open0 = open_hand_score(ll0)
    open1 = open_hand_score(ll1)

    fist0 = fist_score(ll0)
    fist1 = fist_score(ll1)

    # Case 1: hand 0 = open static, hand 1 = fist
    case_a_score = open0 + fist1

    # Case 2: hand 1 = open static, hand 0 = fist
    case_b_score = open1 + fist0

    if case_a_score >= case_b_score:
        static_ll = ll0
        fist_ll = ll1
        static_open = open0
        fist_closed = fist1
    else:
        static_ll = ll1
        fist_ll = ll0
        static_open = open1
        fist_closed = fist0

    # Relaxed requirement
    if static_open < 2:
        return None, None

    if fist_closed < 2:
        return None, None

    return static_ll, fist_ll


def is_movie(hand_res):
    """
    MOVIE gesture:
    - exactly two hands
    - one hand stays open and stable
    - the other hand is a fist
    - the fist moves in a small circular crank/handle motion
    """

    global last_detected_time, last_seen_time

    now = time.time()

    if not hand_res.multi_hand_landmarks or len(hand_res.multi_hand_landmarks) != 2:
        reset_movie()
        return False

    ll0 = hand_res.multi_hand_landmarks[0].landmark
    ll1 = hand_res.multi_hand_landmarks[1].landmark

    static_ll, fist_ll = choose_static_and_fist_hand(ll0, ll1)

    if static_ll is None or fist_ll is None:
        reset_movie()
        return False

    if now - last_seen_time > 0.7:
        reset_movie()

    last_seen_time = now

    static_x, static_y = hand_center(static_ll)
    fist_x, fist_y = hand_center(fist_ll)

    palm_avg = (palm_size(static_ll) + palm_size(fist_ll)) / 2

    history.append((
        now,
        static_x,
        static_y,
        fist_x,
        fist_y,
        palm_avg
    ))

    # Cooldown after successful detection
    if now - last_detected_time < 0.9:
        return False

    if len(history) < 10:
        return False

    start_t = history[0][0]
    end_t = history[-1][0]
    dt = end_t - start_t

    if dt < 0.30 or dt > 3.0:
        return False

    palm = history[-1][5]

    static_xs = [item[1] for item in history]
    static_ys = [item[2] for item in history]

    fist_xs = [item[3] for item in history]
    fist_ys = [item[4] for item in history]

    # Static hand should stay mostly fixed
    static_move_x = (max(static_xs) - min(static_xs)) / palm
    static_move_y = (max(static_ys) - min(static_ys)) / palm

    if static_move_x > 0.55 or static_move_y > 0.55:
        return False

    # Fist must move in both x and y directions
    fist_range_x = (max(fist_xs) - min(fist_xs)) / palm
    fist_range_y = (max(fist_ys) - min(fist_ys)) / palm

    if fist_range_x < 0.12:
        return False

    if fist_range_y < 0.10:
        return False

    # Total fist path length
    total_path = 0

    for i in range(1, len(history)):
        dx = fist_xs[i] - fist_xs[i - 1]
        dy = fist_ys[i] - fist_ys[i - 1]
        total_path += math.sqrt(dx * dx + dy * dy) / palm

    if total_path < 0.55:
        return False

    # Check circular/cranking direction around fist path center
    center_x = sum(fist_xs) / len(fist_xs)
    center_y = sum(fist_ys) / len(fist_ys)

    angles = []

    for x, y in zip(fist_xs, fist_ys):
        angles.append(math.atan2(y - center_y, x - center_x))

    total_angle_change = 0

    for i in range(1, len(angles)):
        total_angle_change += abs(angle_diff(angles[i - 1], angles[i]))

    # Relaxed circular motion check
    if total_angle_change < math.pi * 0.8:
        return False

    last_detected_time = now
    reset_movie()
    return True