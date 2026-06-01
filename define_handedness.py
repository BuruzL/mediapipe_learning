def handedness_label(hand_info):
    """Return the user's actual hand: 'Left' or 'Right'."""
    label = hand_info.classification[0].label
    return label
# returns the handedness of the passed
# hand_info contains .multi_handedness info one hand