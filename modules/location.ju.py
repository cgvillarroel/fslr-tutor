# %%
# Allow imports from another folder
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


# %%
def determineBucket(hand, left_shoulder, right_shoulder):
    if hand.y < left_shoulder.y:

        if hand.x < right_shoulder.x:
            return 0

        if hand.x < left_shoulder.x:
            return 1

        return 2

    if hand.x < right_shoulder.x:
        return 3

    if hand.x < left_shoulder.x:
        return 4

    return 5


# %%
def trackBuckets(clip):
    left_buckets = [0] * 6
    right_buckets = [0] * 6

    for frame in clip.frames:
        # shoulders act as boundary markers for buckets
        left_shoulder = frame.pose_landmarks[11]
        right_shoulder = frame.pose_landmarks[12]
        left_shoulder.y = (left_shoulder.y + right_shoulder.y) / 2
        right_shoulder.y = left_shoulder.y

        left_wrist = frame.pose_landmarks[15]
        right_wrist = frame.pose_landmarks[16]

        # only track when sure to be wrist (avoid phantoms)
        if left_wrist.visibility > 0.9:
            left_idx = determineBucket(left_wrist, left_shoulder, right_shoulder)
            left_buckets[left_idx] += 1
        # if not in frame, assume it's at their side
        # to avoid zero vectors
        else:
            left_buckets[5] += 1

        # only track when sure to be wrist (avoid phantoms)
        if right_wrist.visibility > 0.9:
            right_idx = determineBucket(right_wrist, left_shoulder, right_shoulder)
            right_buckets[right_idx] += 1
        # if not in frame, assume it's at their side
        # to avoid zero vectors
        else:
            right_buckets[3] += 1

    return left_buckets, right_buckets


# %%
from structs.functions import cosine
def compareHandLocations(clip1, clip2):
    buckets1 = trackBuckets(clip1)
    buckets2 = trackBuckets(clip2)

    return cosine(buckets1[0], buckets2[0]), cosine(buckets1[1], buckets2[1])


# %%
# Sample Usage
if __name__ == "__main__":
    import pickle

    data = []

    with open("../dataset/gestures/0.pkl", "rb") as reader:
        data.append(pickle.load(reader))

    with open("../dataset/gestures/6.pkl", "rb") as reader:
        data.append(pickle.load(reader))

    print(compareHandLocations(data[0].clips[0], data[0].clips[1]))
    print(compareHandLocations(data[0].clips[0], data[1].clips[0]))
