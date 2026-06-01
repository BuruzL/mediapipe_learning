import math
def distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def angle(a, b, c):
    v1 = (a.x - b.x, a.y - b.y)
    v2 = (c.x - b.x, c.y - b.y)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag = math.hypot(*v1) * math.hypot(*v2) + 1e-6
    cos_angle = max(-1, min(1, dot / mag))
    return math.degrees(math.acos(cos_angle))

def count_fingers(hand_landmark):
    count = 0
    ll = hand_landmark.landmark
    wrist = ll[0]
    #the thumb is calculated in a different way since the anatomy is a bit different for that one
    #angles are checked of the several joints of the thumb
    if angle(ll[2], ll[3], ll[4]) > 145 and angle(ll[1], ll[2], ll[3]) > 150:
        count += 1
    palm = distance(ll[0], ll[9]) + 1e-6
    for tip, joint in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        if (distance(ll[tip], wrist) - distance(ll[joint], wrist)) / palm > 0.4:
            count += 1
    return count
# a common trick which is independent of distance from the webcam is to use the ratio
# of the several parts of the hand, here palm distance is used to keep track of ratio