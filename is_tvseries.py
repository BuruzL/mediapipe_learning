from collections import deque
import time

# ============================================================
# TVSERIES gesture:
# Two index fingers start together near top-middle
# -> pull outward
# -> move down
# -> bring together at bottom
# ============================================================

phase = "ready"
last_detected_time = 0
last_seen_time = 0
phase_start_time = 0

missing_frames = 0

start_gap = 0
start_left_x = 0
start_right_x = 0
start_y = 0

wide_gap = 0
bottom_y = 0

DEBUG = True


def reset_tvseries():
    global phase, phase_start_time, missing_frames
    global start_gap, start_left_x, start_right_x, start_y
    global wide_gap, bottom_y

    phase = "ready"
    phase_start_time = 0
    missing_frames = 0

    start_gap = 0
    start_left_x = 0
    start_right_x = 0
    start_y = 0

    wide_gap = 0
    bottom_y = 0


def debug_print(msg):
    if DEBUG:
        print("[TVSERIES]", msg)


def set_phase(new_phase):
    global phase

    if phase != new_phase:
        phase = new_phase
        debug_print(f"phase -> {phase}")


def index_is_open(ll):
    wrist = ll[0]
    index_mcp = ll[5]
    index_tip = ll[8]

    tip_dist = ((index_tip.x - wrist.x) ** 2 + (index_tip.y - wrist.y) ** 2) ** 0.5
    mcp_dist = ((index_mcp.x - wrist.x) ** 2 + (index_mcp.y - wrist.y) ** 2) ** 0.5

    return tip_dist > mcp_dist * 1.15


def get_two_hands_sorted(hand_res):
    hands = hand_res.multi_hand_landmarks

    ll0 = hands[0].landmark
    ll1 = hands[1].landmark

    if ll0[0].x < ll1[0].x:
        return ll0, ll1

    return ll1, ll0


def is_tvseries(hand_res):
    global last_detected_time, last_seen_time, phase_start_time
    global missing_frames
    global start_gap, start_left_x, start_right_x, start_y
    global wide_gap, bottom_y

    now = time.time()

    # Cooldown after successful detection
    if now - last_detected_time < 1.0:
        return False

    # Need exactly two hands
    if not hand_res.multi_hand_landmarks or len(hand_res.multi_hand_landmarks) != 2:
        missing_frames += 1

        if missing_frames > 6:
            reset_tvseries()

        return False

    left_ll, right_ll = get_two_hands_sorted(hand_res)

    # Need both index fingers open
    if not index_is_open(left_ll) or not index_is_open(right_ll):
        missing_frames += 1

        if missing_frames > 6:
            reset_tvseries()

        return False

    missing_frames = 0

    if now - last_seen_time > 1.0:
        reset_tvseries()

    last_seen_time = now

    # Index fingertip positions
    left_x = left_ll[8].x
    left_y = left_ll[8].y

    right_x = right_ll[8].x
    right_y = right_ll[8].y

    gap = right_x - left_x
    avg_y = (left_y + right_y) / 2
    center_x = (left_x + right_x) / 2
    y_diff = abs(left_y - right_y)

    # ============================================================
    # Tunable raw screen-coordinate thresholds
    # MediaPipe x/y are normalized from 0 to 1.
    # x increases left -> right
    # y increases top -> bottom
    # ============================================================

    START_MAX_GAP = 0.22
    START_TOP_MAX_Y = 0.55
    START_CENTER_MIN = 0.20
    START_CENTER_MAX = 0.80
    START_MAX_Y_DIFF = 0.18

    OUTWARD_GAP_INCREASE = 0.20
    EACH_HAND_OUTWARD = 0.06

    DOWN_AMOUNT = 0.15
    MIN_WIDE_GAP = 0.28

    INWARD_GAP_DECREASE = 0.18
    FINAL_MAX_GAP = 0.25

    MAX_TOTAL_TIME = 6.0

    # Timeout
    if phase != "ready" and now - phase_start_time > MAX_TOTAL_TIME:
        debug_print("timeout reset")
        reset_tvseries()
        return False

    # ============================================================
    # Phase 1: start with two index fingers close together
    # ============================================================

    if phase == "ready":
        if (
            gap < START_MAX_GAP
            and avg_y < START_TOP_MAX_Y
            and START_CENTER_MIN < center_x < START_CENTER_MAX
            and y_diff < START_MAX_Y_DIFF
        ):
            phase_start_time = now

            start_gap = gap
            start_left_x = left_x
            start_right_x = right_x
            start_y = avg_y

            set_phase("outward")

        return False

    # ============================================================
    # Phase 2: pull fingers outward
    # ============================================================

    if phase == "outward":
        left_outward = start_left_x - left_x
        right_outward = right_x - start_right_x
        gap_increase = gap - start_gap

        if (
            gap_increase > OUTWARD_GAP_INCREASE
            and left_outward > EACH_HAND_OUTWARD
            and right_outward > EACH_HAND_OUTWARD
            and y_diff < 0.25
        ):
            wide_gap = gap
            set_phase("down")

        return False

    # ============================================================
    # Phase 3: move both fingers downward
    # ============================================================

    if phase == "down":
        downward_move = avg_y - start_y

        if (
            downward_move > DOWN_AMOUNT
            and gap > MIN_WIDE_GAP
            and y_diff < 0.30
        ):
            bottom_y = avg_y
            set_phase("inward")

        return False

    # ============================================================
    # Phase 4: bring fingers inward together at bottom
    # ============================================================

    if phase == "inward":
        gap_decrease = wide_gap - gap
        still_low = avg_y > start_y + DOWN_AMOUNT * 0.65

        if (
            gap_decrease > INWARD_GAP_DECREASE
            and gap < FINAL_MAX_GAP
            and still_low
            and y_diff < 0.30
        ):
            debug_print("DETECTED")
            last_detected_time = now
            reset_tvseries()
            return True

        return False

    return False