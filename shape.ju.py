# %%
from structs.types import Gesture, Clip, NormalizedLandmark
def extractLeftHandShapes(clip: Clip):
    features = []
    samples_per_clip = 6
    interval = len(clip.frames) // samples_per_clip

    for frame in clip.frames[:interval * samples_per_clip:interval]:
        if (frame.pose_landmarks[15].visibility < 0.9
                or len(frame.left_hand_landmarks) == 0):
            features.extend([1] * 42)

        features.extend([[a.x, a.y] for a in frame.left_hand_landmarks])

    return features

def extractRightHandShapes(clip: Clip):
    features = []
    samples_per_clip = 6
    interval = len(clip.frames) // samples_per_clip

    for frame in clip.frames[:interval * samples_per_clip:interval]:
        if (frame.pose_landmarks[16].visibility < 0.9
                or len(frame.right_hand_landmarks) == 0):
            features.extend([1] * 42)

        _ = [features.extend([a.x, a.y]) for a in frame.right_hand_landmarks]

    return features

# %%
import math
def euclideanDistance(point1: NormalizedLandmark, point2: NormalizedLandmark):
    x = point2.x - point1.x
    y = point2.y - point1.y
    return math.sqrt((x ** 2) + (y ** 2))

def compareHandShapes(clip1: Clip, clip2: Clip):
    left_error = 0
    right_error = 0

    left_samples1 = []
    left_samples2 = []
    right_samples1 = []
    right_samples2 = []

    samples_per_clip = 6
    interval1 = len(clip1.frames) // samples_per_clip
    interval2 = len(clip2.frames) // samples_per_clip

    # transform into a more manageable structure (array of points)
    for frame in clip1.frames[:interval1 * samples_per_clip:interval1]:
        if (frame.pose_landmarks[15].visibility < 0.9
                or len(frame.left_hand_landmarks) == 0):
            left_samples1.append([NormalizedLandmark(0.0, 0.0, 0.0)] * 42)

        left_samples1.append(frame.left_hand_landmarks)

    for frame in clip2.frames[:interval2 * samples_per_clip:interval2]:
        if (frame.pose_landmarks[15].visibility < 0.9
                or len(frame.left_hand_landmarks) == 0):
            left_samples2.append([NormalizedLandmark(0.0, 0.0, 0.0)] * 42)

        left_samples2.append(frame.left_hand_landmarks)

    for frame in clip1.frames[:interval1 * samples_per_clip:interval1]:
        if (frame.pose_landmarks[16].visibility < 0.9
                or len(frame.right_hand_landmarks) == 0):
            right_samples1.append([NormalizedLandmark(0.0, 0.0, 0.0)] * 42)

        right_samples1.append(frame.right_hand_landmarks)

    for frame in clip2.frames[:interval2 * samples_per_clip:interval2]:
        if (frame.pose_landmarks[16].visibility < 0.9
                or len(frame.right_hand_landmarks) == 0):
            right_samples2.append([NormalizedLandmark(0.0, 0.0, 0.0)] * 42)

        right_samples2.append(frame.right_hand_landmarks)

    # compute total error
    for left_sample1, left_sample2 in zip(left_samples1, left_samples2):
        for point1, point2 in zip(left_sample1, left_sample2):
            left_error += euclideanDistance(point1, point2)

    for right_sample1, right_sample2 in zip(right_samples1, right_samples2):
        for point1, point2 in zip(right_sample1, right_sample2):
            right_error += euclideanDistance(point1, point2)

    return left_error, right_error

# %%
def compareFaces(clip1: Clip, clip2: Clip):
    error = 0

    samples1 = []
    samples2 = []

    samples_per_clip = 6
    interval1 = len(clip1.frames) // samples_per_clip
    interval2 = len(clip2.frames) // samples_per_clip

    # transform into a more manageable structure (array of points)
    for frame in clip1.frames[:interval1 * samples_per_clip:interval1]:
        samples1.append(frame.face_landmarks)

    for frame in clip2.frames[:interval2 * samples_per_clip:interval2]:
        samples2.append(frame.face_landmarks)

    # compute total error
    for sample1, sample2 in zip(samples1, samples2):
        for point1, point2 in zip(sample1, sample2):
            error += euclideanDistance(point1, point2)

    return error

# %%
import pickle
from structs.functions import cosine

data: list[Gesture] = []

with open("dataset/gestures/0.pkl", "rb") as reader:
    data.append(pickle.load(reader))

with open("dataset/gestures/1.pkl", "rb") as reader:
    data.append(pickle.load(reader))

# %%
left_shapes = []
left_shapes.append(extractLeftHandShapes(data[0].clips[0]))
left_shapes.append(extractLeftHandShapes(data[0].clips[1]))
left_shapes.append(extractLeftHandShapes(data[1].clips[0]))

right_shapes = []
right_shapes.append(extractRightHandShapes(data[0].clips[0]))
right_shapes.append(extractRightHandShapes(data[0].clips[1]))
right_shapes.append(extractRightHandShapes(data[1].clips[0]))

print(cosine(left_shapes[0], left_shapes[1]))
print(cosine(right_shapes[0], right_shapes[1]))
print(cosine(left_shapes[0], left_shapes[2]))
print(cosine(right_shapes[0], right_shapes[2]))

# %%
print(compareHandShapes(data[0].clips[0], data[0].clips[1]))
print(compareHandShapes(data[0].clips[0], data[1].clips[0]))

# %%
print(compareFaces(data[0].clips[0], data[0].clips[1]))
print(compareFaces(data[0].clips[0], data[1].clips[0]))
