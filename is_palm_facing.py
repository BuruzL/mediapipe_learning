def is_palm_facing(hand_landmark, label):
    ll = hand_landmark.landmark
    thumb_tip  = ll[4]
    pinky_base = ll[17]
    thumb_left_of_pinky = thumb_tip.x < pinky_base.x

    if label == "Right":
        return thumb_left_of_pinky
    else:
        return not thumb_left_of_pinky
# one thing to be noted: label must be passed to the function
# label states the handedness of the passed hand