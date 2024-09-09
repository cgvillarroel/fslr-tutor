# %%
# Allow imports from another folder
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# %%
from structs.types import Gesture, Clip
from structs.functions import euclideanDistance, cosine

def compareFacesEuclid(clip1: Clip, clip2: Clip):
    error = 0

    samples1 = []
    samples2 = []

    samples_per_clip = 6
    interval1 = len(clip1.frames) // samples_per_clip
    interval2 = len(clip2.frames) // samples_per_clip

    # transform into a more manageable structure (array of points)
    for frame in clip1.frames[:interval1 * samples_per_clip:interval1]:
        samples1.append(frame.pose_landmarks[:11])

    for frame in clip2.frames[:interval2 * samples_per_clip:interval2]:
        samples2.append(frame.pose_landmarks[:11])

    # compute total error
    for sample1, sample2 in zip(samples1, samples2):
        for point1, point2 in zip(sample1, sample2):
            error += euclideanDistance(point1, point2)

    return error


def compareFacesCosine(clip1: Clip, clip2: Clip):
    samples1 = []
    samples2 = []

    samples_per_clip = 6
    interval1 = len(clip1.frames) // samples_per_clip
    interval2 = len(clip2.frames) // samples_per_clip

    # transform into a more manageable structure (array of points)
    for frame in clip1.frames[:interval1 * samples_per_clip:interval1]:
        _ = [samples1.extend([a.x, a.y]) for a in frame.pose_landmarks[:11]]

    for frame in clip2.frames[:interval2 * samples_per_clip:interval2]:
        _ = [samples2.extend([a.x, a.y]) for a in frame.pose_landmarks[:11]]

    return cosine(samples1, samples2)


# %%
# Sample usage
if __name__ == "__main__":
    import pickle

    data: list[Gesture] = []

    with open("../dataset/gestures/0.pkl", "rb") as reader:
        data.append(pickle.load(reader))

    with open("../dataset/gestures/1.pkl", "rb") as reader:
        data.append(pickle.load(reader))

    print(compareFacesCosine(data[0].clips[0], data[0].clips[1]))
    print(compareFacesCosine(data[0].clips[0], data[1].clips[0]))
