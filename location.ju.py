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
import pickle
from structs.functions import cosine

data = []

with open("dataset/gestures/0.pkl", "rb") as reader:
    data.append(pickle.load(reader))

with open("dataset/gestures/6.pkl", "rb") as reader:
    data.append(pickle.load(reader))

buckets_a = trackBuckets(data[0].clips[0])
buckets_b = trackBuckets(data[0].clips[1])
buckets_c = trackBuckets(data[1].clips[0])

print(buckets_a)
print(buckets_b)
print(buckets_c)

print(cosine(buckets_a[0], buckets_b[0]))
print(cosine(buckets_a[1], buckets_b[1]))
print(cosine(buckets_a[0], buckets_c[0]))
print(cosine(buckets_a[1], buckets_c[1]))
