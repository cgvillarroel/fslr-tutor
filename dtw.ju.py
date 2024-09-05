# %%
import numpy as np

# %%
# https://github.com/kamperh/lecture_dtw_notebook/blob/main/dtw.ipynb


def dtw(x, y, cost):
    N = x.shape[0]
    M = y.shape[0]
    dist_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            dist_mat[i, j] = cost(x[i], y[j])

    N, M = dist_mat.shape

    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]]  # deletion (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    # Traceback from bottom right
    i = N - 1
    j = M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # Match
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            # Insertion
            i = i - 1
        elif tb_type == 2:
            # Deletion
            j = j - 1
        path.append((i, j))

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    # return (path[::-1], cost_mat)

    return cost_mat[N - 1, M - 1], cost_mat[N - 1, M - 1] / (N + M)


# %%
from structs.functions import cosine
from structs.types import Gesture, Frame


# will 
def cost_function_generator(landmark_idx):
    def cost(frame1: Frame, frame2: Frame):
        frame1_point = frame1.pose_landmarks[landmark_idx]
        frame2_point = frame2.pose_landmarks[landmark_idx]
        return cosine([frame1_point.x, frame1_point.y], [frame2_point.x, frame2_point.y])

    return cost


# %%
import pickle

data: list[Gesture] = []

with open("dataset/gestures/0.pkl", "rb") as reader:
    data.append(pickle.load(reader))

with open("dataset/gestures/6.pkl", "rb") as reader:
    data.append(pickle.load(reader))

series1 = data[0].clips[0].frames
series2 = data[1].clips[0].frames

# %%
for i in range(11, 17):
    result = dtw(np.array(series1), np.array(series2), cost_function_generator(i))
    print(f"{i}: {result[0]:.5f} (total), {result[1]:.5f} (normalized)")
