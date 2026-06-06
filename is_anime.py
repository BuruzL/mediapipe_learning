from count_fingers import distance


def palm_size(ll):
    return distance(ll[0], ll[9]) + 1e-6


def hand_center(ll):
    x = (ll[0].x + ll[5].x + ll[9].x + ll[13].x + ll[17].x) / 5
    y = (ll[0].y + ll[5].y + ll[9].y + ll[13].y + ll[17].y) / 5
    return x, y


def fingertips_center(ll):
    x = (ll[8].x + ll[12].x + ll[16].x + ll[20].x) / 4
    y = (ll[8].y + ll[12].y + ll[16].y + ll[20].y) / 4
    return x, y


def open_flat_hand_score(ll):
    """
    Detects open flat hand:
    - index, middle, ring, pinky extended
    - fingers are not curled
    """

    palm = palm_size(ll)

    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    mcp_ids = [5, 9, 13, 17]

    score = 0

    for tip, pip, mcp in zip(tip_ids, pip_ids, mcp_ids):
        tip_to_wrist = distance(ll[tip], ll[0])
        pip_to_wrist = distance(ll[pip], ll[0])
        tip_to_mcp = distance(ll[tip], ll[mcp])

        if tip_to_wrist > pip_to_wrist + 0.05 * palm and tip_to_mcp > 0.55 * palm:
            score += 1

    return score


def fingers_together(ll):
    """
    In the reference pose, fingers are mostly together,
    not spread like a V sign.
    """

    palm = palm_size(ll)

    gap_index_middle = distance(ll[8], ll[12]) / palm
    gap_middle_ring = distance(ll[12], ll[16]) / palm
    gap_ring_pinky = distance(ll[16], ll[20]) / palm

    if gap_index_middle > 0.75:
        return False

    if gap_middle_ring > 0.75:
        return False

    if gap_ring_pinky > 0.85:
        return False

    return True


def hand_points_inward(ll, side):
    """
    side = "left" or "right" based on screen position.

    For the kawaii chin pose:
    - left hand fingertips point right
    - right hand fingertips point left
    """

    palm = palm_size(ll)

    wrist_x = ll[0].x
    wrist_y = ll[0].y

    tip_x, tip_y = fingertips_center(ll)

    dx = (tip_x - wrist_x) / palm
    dy = abs(tip_y - wrist_y) / palm

    if side == "left":
        if dx < 0.35:
            return False

    elif side == "right":
        if dx > -0.35:
            return False

    # Hand should be mostly sideways/horizontal, not straight vertical
    if dy > 1.20:
        return False

    return True


def hand_near_chin_area(ll):
    """
    Approximation only.

    Current project uses hand landmarks only, not face/chin landmarks.
    So we approximate chin area as upper-middle screen area.
    """

    cx, cy = hand_center(ll)
    tip_x, tip_y = fingertips_center(ll)

    # Not too low, not too high
    if cy < 0.18 or cy > 0.75:
        return False

    if tip_y < 0.18 or tip_y > 0.72:
        return False

    # Hands should be around the body/face area
    if cx < 0.05 or cx > 0.95:
        return False

    # Fingertips should not be at the bottom of the frame
    if tip_x < 0.10 or tip_x > 0.90:
        return False

    return True


def two_open_hands_under_chin(ll0, ll1):
    x0, y0 = hand_center(ll0)
    x1, y1 = hand_center(ll1)

    # Sort hands by screen position
    if x0 <= x1:
        left_ll = ll0
        right_ll = ll1
    else:
        left_ll = ll1
        right_ll = ll0

    left_x, left_y = hand_center(left_ll)
    right_x, right_y = hand_center(right_ll)

    left_tip_x, left_tip_y = fingertips_center(left_ll)
    right_tip_x, right_tip_y = fingertips_center(right_ll)

    palm_avg = (palm_size(left_ll) + palm_size(right_ll)) / 2

    # Both hands must be open flat hands
    if open_flat_hand_score(left_ll) < 3:
        return False

    if open_flat_hand_score(right_ll) < 3:
        return False

    # Fingers should be together, not V-shaped
    if not fingers_together(left_ll):
        return False

    if not fingers_together(right_ll):
        return False

    # Both hands should be in approximate chin/face region
    if not hand_near_chin_area(left_ll):
        return False

    if not hand_near_chin_area(right_ll):
        return False

    # Left hand points inward, right hand points inward
    if not hand_points_inward(left_ll, "left"):
        return False

    if not hand_points_inward(right_ll, "right"):
        return False

    # Hands should be roughly at the same height
    hand_y_gap = abs(left_y - right_y) / palm_avg

    if hand_y_gap > 0.90:
        return False

    # Fingertips should also be at similar height
    tip_y_gap = abs(left_tip_y - right_tip_y) / palm_avg

    if tip_y_gap > 0.75:
        return False

    # Fingertips should be closer to the center than wrists
    left_wrist_x = left_ll[0].x
    right_wrist_x = right_ll[0].x

    if abs(left_tip_x - 0.5) > abs(left_wrist_x - 0.5):
        return False

    if abs(right_tip_x - 0.5) > abs(right_wrist_x - 0.5):
        return False

    # The two fingertip groups should be under the chin, near center
    tip_pair_center_x = (left_tip_x + right_tip_x) / 2
    tip_pair_center_y = (left_tip_y + right_tip_y) / 2

    if tip_pair_center_x < 0.30 or tip_pair_center_x > 0.70:
        return False

    if tip_pair_center_y < 0.20 or tip_pair_center_y > 0.70:
        return False

    # Hands should not be too far apart
    hand_gap = abs(left_x - right_x) / palm_avg

    if hand_gap < 0.80:
        return False

    if hand_gap > 4.00:
        return False

    # Fingertips should be closer than wrists
    wrist_gap = abs(left_wrist_x - right_wrist_x)
    tip_gap = abs(left_tip_x - right_tip_x)

    if tip_gap > wrist_gap:
        return False

    return True


def is_anime(hand_res):
    """
    ANIME / KAWAII gesture:

    Pose like the reference image:
    - exactly two hands visible
    - both hands open and flat
    - fingers together
    - hands under chin area
    - left hand points inward
    - right hand points inward
    """

    if not hand_res.multi_hand_landmarks:
        return False

    if len(hand_res.multi_hand_landmarks) != 2:
        return False

    ll0 = hand_res.multi_hand_landmarks[0].landmark
    ll1 = hand_res.multi_hand_landmarks[1].landmark

    return two_open_hands_under_chin(ll0, ll1)