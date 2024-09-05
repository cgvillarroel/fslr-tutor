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
