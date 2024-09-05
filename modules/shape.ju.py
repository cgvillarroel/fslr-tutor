# %%
# Allow imports from another folder
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


# %%
from structs.types import Gesture, Clip, NormalizedLandmark
from structs.functions import euclideanDistance

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
# Sample usage
if __name__ == "__main__":
    import pickle

    data: list[Gesture] = []

    with open("../dataset/gestures/0.pkl", "rb") as reader:
        data.append(pickle.load(reader))

    with open("../dataset/gestures/1.pkl", "rb") as reader:
        data.append(pickle.load(reader))

    print(compareHandShapes(data[0].clips[0], data[0].clips[1]))
    print(compareHandShapes(data[0].clips[0], data[1].clips[0]))
