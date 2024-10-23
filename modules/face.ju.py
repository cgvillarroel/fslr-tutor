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


def extractKeyLandmarks(face_landmarks):
    return [
        face_landmarks[336], # left inner eyebrow
        face_landmarks[334], # left middle eyebrow
        face_landmarks[300], # left outer eyebrow

        face_landmarks[107], # right inner eyebrow
        face_landmarks[105], # right middle eyebrow
        face_landmarks[70], # right outer eyebrow

        face_landmarks[386], # left upper eyelid
        face_landmarks[374], # left lower eyelid

        face_landmarks[159], # right upper eyelid
        face_landmarks[145], # right lower eyelid

        face_landmarks[291], # left lip
        face_landmarks[61], # right lip

        face_landmarks[0], # outer upper lip
        face_landmarks[13], # inner upper lip
        face_landmarks[14], # inner lower lip
        face_landmarks[17], # outer lower lip
    ]


def compareFacesEuclid(clip1: Clip, clip2: Clip):
    error = 0

    samples1 = []
    samples2 = []

    samples_per_clip = 6
    samples_per_clip += 1
    interval1 = len(clip1.frames) // samples_per_clip
    interval2 = len(clip2.frames) // samples_per_clip

    # transform into a more manageable structure (array of points)
    for frame in clip1.frames[interval1 : interval1 * samples_per_clip : interval1]:
        samples1.append(extractKeyLandmarks(frame.face_landmarks))

    for frame in clip2.frames[interval2 : interval2 * samples_per_clip : interval2]:
        samples2.append(extractKeyLandmarks(frame.face_landmarks))

    # compute total error
    for sample1, sample2 in zip(samples1, samples2):
        for point1, point2 in zip(sample1, sample2):
            error += euclideanDistance(point1, point2)

    return error


def compareFacesCosine(clip1: Clip, clip2: Clip):
    samples1 = []
    samples2 = []

    samples_per_clip = 6
    samples_per_clip += 1
    interval1 = len(clip1.frames) // samples_per_clip
    interval2 = len(clip2.frames) // samples_per_clip

    # transform into a more manageable structure (array of points)
    for frame in clip1.frames[interval1 : interval1 * samples_per_clip : interval1]:
        _ = [samples1.extend([a.x, a.y]) for a in extractKeyLandmarks(frame.face_landmarks)]

    for frame in clip2.frames[interval2 : interval2 * samples_per_clip : interval2]:
        _ = [samples2.extend([a.x, a.y]) for a in extractKeyLandmarks(frame.face_landmarks)]

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

    print(compareFacesEuclid(data[0].clips[0], data[0].clips[1]))
    print(compareFacesEuclid(data[0].clips[0], data[1].clips[0]))
