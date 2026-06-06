from count_fingers import *

def is_gun(hand_res):
    if not hand_res.multi_hand_landmarks or len(hand_res.multi_hand_landmarks) != 1:
        return False

    hand_a = hand_res.multi_hand_landmarks[0]
    ll = hand_a.landmark

    palm = distance(ll[0], ll[9]) + 1e-6

    # Thumb open thakbe
    if not (angle(ll[2], ll[3], ll[4]) > 145 and angle(ll[1], ll[2], ll[3]) > 150):
        return False

    # Ring and pinky close thakbe
    for tip, joint in [(16, 13), (20, 17)]:
        if (distance(ll[tip], ll[0]) - distance(ll[joint], ll[0])) / palm > 0.4:
            return False

    # Index and middle open thakbe
    for tip, joint in [(8, 5), (12, 9)]:
        if (distance(ll[tip], ll[0]) - distance(ll[joint], ll[0])) / palm < 0.4:
            return False

    # Direction check
    index_horizontal = abs(ll[8].x - ll[5].x) > abs(ll[8].y - ll[5].y)
    middle_horizontal = abs(ll[12].x - ll[9].x) > abs(ll[12].y - ll[9].y)

    if index_horizontal and middle_horizontal:
        return True

    return False