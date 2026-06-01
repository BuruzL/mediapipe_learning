from count_fingers import *

def is_bird(hand_res, palm_a, palm_b):

    #both hands must be used hence the length check
    if not hand_res.multi_hand_landmarks or len(hand_res.multi_hand_landmarks) != 2:
        return False
    #assign to variables the landmarks
    a, b = hand_res.multi_hand_landmarks
    la, lb = a.landmark, b.landmark

    if not (palm_a and palm_b): # we check whether both palms are facing towards the webcam
        return False
    if count_fingers(a) != 5 or count_fingers(b) != 5: #all fingers must be used
        return False

    palm = distance(la[0], la[9]) + 1e-6 #calculate the length of the palm (wrist distance to middle finger base)
    thumb_gap    = distance(la[4],  lb[4])  # calculate the distance of the two thumbs
    pinky_spread = distance(la[20], lb[20]) # calculate the distance of the two pinkies
    return thumb_gap < 0.4 * palm and pinky_spread > 2 * palm 
    # an arbitrary calculation that somehow works