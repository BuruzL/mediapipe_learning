from count_fingers import count_fingers
from count_fingers import *
def is_word_ok(hand_res):
    if not hand_res.multi_hand_landmarks or len(hand_res.multi_hand_landmarks)!=2:
        return False
    lhand, rhand = hand_res.multi_hand_landmarks
    ll1 , ll2 = lhand.landmark, rhand.landmark
    if count_fingers(lhand) ==1 and count_fingers(rhand) ==1:
        pass
    else:
        return False
    count =0
    if angle(ll1[2], ll1[3], ll1[4]) > 145 and angle(ll1[1], ll1[2], ll1[3]) > 150:
        count += 1
    if angle(ll2[2], ll2[3], ll2[4]) > 145 and angle(ll2[1], ll2[2], ll2[3]) > 150:
        count += 1
    if count!=2:
        return False
    knuckles1 = [ll1[5].y,ll1[9].y,ll1[13].y,ll1[17].y]
    knuckles2 =[ll2[5].y,ll2[9].y,ll2[13].y,ll2[17].y]
    average1 = sum(knuckles1)/len(knuckles1)
    average2 = sum(knuckles2)/len(knuckles2)
    if average1>ll1[4].y and average2>ll2[4].y:
        return True
    else:
        return False