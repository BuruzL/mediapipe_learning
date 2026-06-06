from count_fingers import distance


def palm_size(ll):
    return distance(ll[0], ll[9]) + 1e-6


def hand_center(ll):
    x = (ll[0].x + ll[5].x + ll[9].x + ll[13].x + ll[17].x) / 5
    y = (ll[0].y + ll[5].y + ll[9].y + ll[13].y + ll[17].y) / 5
    return x, y


def open_finger_count(ll):
    """
    Counts open fingers except thumb.
    """
    palm = palm_size(ll)

    finger_pairs = [
        (8, 5),    # index
        (12, 9),   # middle
        (16, 13),  # ring
        (20, 17),  # pinky
    ]

    count = 0

    for tip, base in finger_pairs:
        openness = (distance(ll[tip], ll[0]) - distance(ll[base], ll[0])) / palm

        if openness > 0.18:
            count += 1

    return count


def is_near_ear_area(ll):
    """
    Hands-only approximation:
    Ear area is assumed to be upper-left or upper-right side of the screen.
    """

    cx, cy = hand_center(ll)

    near_upper_area = cy < 0.62
    near_side_area = cx < 0.42 or cx > 0.58

    return near_upper_area and near_side_area


def is_extended_outward_hand(ll):
    """
    Other hand should look open/extended outward.
    """

    cx, cy = hand_center(ll)
    open_count = open_finger_count(ll)

    # Open hand
    if open_count < 3:
        return False

    # Extended outward usually appears away from screen center
    outward_side = cx < 0.38 or cx > 0.62

    # Not too low
    reasonable_height = cy < 0.85

    return outward_side and reasonable_height


def is_song(hand_res):
    """
    SONG gesture:
    - exactly two hands
    - one hand near ear area
    - other hand extended outward/open
    """

    if not hand_res.multi_hand_landmarks or len(hand_res.multi_hand_landmarks) != 2:
        return False

    ll0 = hand_res.multi_hand_landmarks[0].landmark
    ll1 = hand_res.multi_hand_landmarks[1].landmark

    # Case 1: hand 0 near ear, hand 1 extended
    case_a = is_near_ear_area(ll0) and is_extended_outward_hand(ll1)

    # Case 2: hand 1 near ear, hand 0 extended
    case_b = is_near_ear_area(ll1) and is_extended_outward_hand(ll0)

    return case_a or case_b