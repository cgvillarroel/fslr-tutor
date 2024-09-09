import math

def cosine(vec1, vec2):
    vec_len = len(vec1)
    sum_of_prod = 0
    sum_of_square_1 = 0
    sum_of_square_2 = 0
    for i in range(vec_len):
        sum_of_prod += vec1[i] * vec2[i]
        sum_of_square_1 += vec1[i] * vec1[i]
        sum_of_square_2 += vec2[i] * vec2[i]

    return sum_of_prod / (math.sqrt(sum_of_square_1) * (math.sqrt(sum_of_square_2)))


def euclideanDistance(point1, point2):
    x = point2.x - point1.x
    y = point2.y - point1.y
    return math.sqrt((x ** 2) + (y ** 2))

def accuracy(mat):
    return (mat[0] + mat[1]) / sum(mat)

def precision(mat):
    all_positive = mat[0] + mat[2]
    return mat[0] / all_positive if all_positive > 0 else 0

def recall(mat):
    all_true = mat[0] + mat[1]
    return mat[0] / all_true if all_true > 0 else 0

def f1Score(mat):
    denominator = (2 * mat[0] + mat[2] + mat[3])
    return (2 * mat[0]) / denominator if denominator > 0 else 0

def printStats(mat):
    print(f"TP: {mat[0]}\t FN: {mat[3]}")
    print(f"FP: {mat[2]}\t TN: {mat[1]}\n")

    print(f"Accuracy : {accuracy(mat)}")
    print(f"Precision: {precision(mat)}")
    print(f"Recall   : {recall(mat)}")
    print(f"F1 Score : {f1Score(mat)}")
