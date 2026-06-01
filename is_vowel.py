from models import count_fingers
import math
def distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
def is_vowel(hand_res, palm_a, palm_b):
    # same as is_bird except both palms must be opposite to the webcam
    if not hand_res.multi_hand_landmarks or len(hand_res.multi_hand_landmarks) != 2:
        return False
    a, b = hand_res.multi_hand_landmarks
    la, lb = a.landmark, b.landmark

    if palm_a or palm_b:
        return False
    if count_fingers(a) != 5 or count_fingers(b) != 5:
        return False

    palm = distance(la[0], la[9]) + 1e-6
    thumb_gap    = distance(la[4],  lb[4])
    pinky_spread = distance(la[20], lb[20])
    return thumb_gap < 0.4 * palm and pinky_spread > 2 * palm