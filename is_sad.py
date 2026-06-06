def dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def is_sad(face_res):
    """
    SAD:
    Detects upside-down sad lips / frown-like mouth shape
    using FaceMesh landmarks.

    Landmark idea:
    - 61  = left mouth corner
    - 291 = right mouth corner
    - 13  = upper inner lip
    - 14  = lower inner lip
    - 234 = left face side
    - 454 = right face side

    MediaPipe y-axis:
    - smaller y = higher
    - larger y = lower

    For sad/frown:
    mouth corners go lower than the mouth center.
    """

    if not face_res.multi_face_landmarks:
        return False

    face = face_res.multi_face_landmarks[0]
    lm = face.landmark

    left_corner = lm[61]
    right_corner = lm[291]

    upper_lip = lm[13]
    lower_lip = lm[14]

    left_face = lm[234]
    right_face = lm[454]

    face_width = dist(left_face, right_face) + 1e-6

    mouth_width = dist(left_corner, right_corner) / face_width
    mouth_open = dist(upper_lip, lower_lip) / face_width

    mouth_center_y = (upper_lip.y + lower_lip.y) / 2
    corner_y = (left_corner.y + right_corner.y) / 2

    # Positive means mouth corners are lower than mouth center
    frown_curve = (corner_y - mouth_center_y) / face_width

    # Tune these if needed
    MAX_MOUTH_WIDTH = 0.42
    MAX_MOUTH_OPEN = 0.12
    FROWN_CURVE_THRESHOLD = 0.010

    if (
        mouth_width < MAX_MOUTH_WIDTH
        and mouth_open < MAX_MOUTH_OPEN
        and frown_curve > FROWN_CURVE_THRESHOLD
    ):
        return True

    return False