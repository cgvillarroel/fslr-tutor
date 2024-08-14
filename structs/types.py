class NormalizedLandmark:
    def __init__(self, x: float, y: float, visibility: float) -> None:
        self.x = x
        self.y = y
        self.visibility = visibility


class Frame:
    def __init__(self):
        self.pose_landmarks: list[NormalizedLandmark] = []
        self.face_landmarks: list[NormalizedLandmark] = []
        self.left_hand_landmarks: list[NormalizedLandmark] = []
        self.right_hand_landmarks: list[NormalizedLandmark] = []

class Clip:
    def __init__(self):
        self.frames: list[Frame] = []


class Gesture:
    def __init__(self):
        self.clips: list[Clip] = []
