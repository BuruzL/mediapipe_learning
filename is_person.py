from count_fingers import *
def is_person(hand_res):
    if not hand_res.multi_hand_landmarks or len(hand_res.multi_hand_landmarks)!=1:
        return False
    hand_a = hand_res.multi_hand_landmarks[0]
    ll = hand_a.landmark
    palm = distance(ll[0], ll[9]) + 1e-6
    for tip, joint in [(16, 13), (20, 17)]:
        if (distance(ll[tip], ll[0]) - distance(ll[joint], ll[0])) / palm > 0.4:
            return False
    for tip, joint in [(8, 5), (12, 9)]:
        if (distance(ll[tip], ll[0]) - distance(ll[joint], ll[0])) / palm < 0.4:
         return False
    if ll[5].y < ll[8].y and ll[9].y < ll[12].y:
        return True
    return False