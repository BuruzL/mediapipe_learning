from count_fingers import *
def is_drunk(hand_res):
    if not hand_res.multi_hand_landmarks or len(hand_res.multi_hand_landmarks)!= 1:
        return False
    hand = hand_res.multi_hand_landmarks[0]
    ll = hand.landmark
    count =0 
    if count_fingers(hand)!=2:
        return False
    if angle(ll[2], ll[3], ll[4]) > 145 and angle(ll[1], ll[2], ll[3]) > 150:
        count += 1
    palm = distance(ll[0], ll[9]) + 1e-6
    for tip, joint in [(8, 5)]:
        if (distance(ll[tip], ll[0]) - distance(ll[joint], ll[0])) / palm > 0.4:
            count += 1
    if(count!=2):
        return False
    if ll[4].y <= ll[17].y:
        return False
    return True
    