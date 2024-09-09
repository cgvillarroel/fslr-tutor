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


class Result:
    def __init__(self, gesture1=0, gesture2=0, location_results=None, motion_results=None, shape_results=None, face_result=None):
        self.gesture1: int = gesture1
        self.gesture2: int = gesture2
        self.location_results: list[float] = location_results if location_results else [0.0]
        self.motion_results: list[float] = motion_results if motion_results else [0.0]
        self.shape_results: list[float] = shape_results if shape_results else [0.0]
        self.face_result: float = face_result if face_result else 0.0
