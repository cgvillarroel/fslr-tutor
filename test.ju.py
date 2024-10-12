# %%
# import data
import pickle
import random
from structs.types import Result

results: list[Result] = []

for i in range(16):
    with open(f"results/{i}.pkl", "rb") as reader:
        results.extend(pickle.load(reader))

print(len(results))

feedback_results: list[Result] = []

for i in range(5):
    with open(f"results/feedback_imbalanced/{i}.pkl", "rb") as reader:
        feedback_results.extend(pickle.load(reader))

print(len(feedback_results))

# %%
# show sample data
print("==========")
print(f"gesture {results[0].gesture1} vs gesture {results[0].gesture2}")
print("----------")
print("location similarity:")
print(f"\tleft hand : {results[0].location_results[0]:.5f}")
print(f"\tright hand: {results[0].location_results[1]:.5f}")
print("motion error:")
print(f"\tleft shoulder : {results[0].motion_results[0]:.5f}")
print(f"\tright shoulder: {results[0].motion_results[1]:.5f}")
print(f"\tleft elbow    : {results[0].motion_results[2]:.5f}")
print(f"\tright elbow   : {results[0].motion_results[3]:.5f}")
print(f"\tleft wrist    : {results[0].motion_results[4]:.5f}")
print(f"\tright wrist   : {results[0].motion_results[5]:.5f}")
print("shape error:")
print(f"\tleft hand : {results[0].shape_results[0]:.5f}")
print(f"\tright hand: {results[0].shape_results[1]:.5f}")


# %% [md]
# # Finding thresholds

# Now that we've run our dataset through the modules, now we have to actually see what values would give us the best stats


# %%
# given the processed results, find the stats given a threshold
def thresholdTesterFactory(false_condition):

    def testThreshold(results: list[Result], threshold: float):
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0

        for result in results:
            if (false_condition(result, threshold)):
                if result.gesture1 == result.gesture2:
                    false_neg += 1
                    continue

                true_neg += 1
                continue

            if result.gesture1 == result.gesture2:
                true_pos += 1
                continue

            false_pos += 1

        return [true_pos, true_neg, false_pos, false_neg]

    return testThreshold


# %%
# find optimal threshold
import matplotlib.pyplot as plt
import numpy as np
import structs.functions as utils


def plotThresholds(test_func, start, end, scale=1000, title="Thresholds", results=results):
    thresholds = [i / scale for i in range(start, end)]
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        print(f"threshold: {threshold}  ", end="\r")

        confusion_matrix = test_func(results, threshold)
        accuracy = utils.accuracy(confusion_matrix)
        precision = utils.precision(confusion_matrix)
        recall = utils.recall(confusion_matrix)
        f1_score = utils.f1Score(confusion_matrix)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    np_thresholds = np.array(thresholds)
    np_accuracies = np.array(accuracies)
    np_precisions = np.array(precisions)
    np_recalls = np.array(recalls)
    np_f1_scores = np.array(f1_scores)

    plt.plot(np_thresholds, np_accuracies, color="r", label="accuracy")
    plt.plot(np_thresholds, np_precisions, color="g", label="precision")
    plt.plot(np_thresholds, np_recalls, color="b", label="recall")
    plt.plot(np_thresholds, np_f1_scores, color="y", label="f1 score")

    plt.title(title)
    plt.legend()
    plt.show()


# %% [md]
# ## Location


# %%
def locationCondition(result, threshold):
    return result.location_results[0] < threshold or result.location_results[1] < threshold


test_location = thresholdTesterFactory(locationCondition)


# %%
plotThresholds(test_location, 825, 1000, title="Location thresholds")


# %%
location_threshold = 0.975
print(f"Threshold: {location_threshold}")
print("==========")
confusion_matrix = test_location(results, location_threshold)
utils.printStats(confusion_matrix)


# %% [md]
# ## Motion


# %%
def motionCondition(result, threshold):
    return (result.motion_results[0] > threshold
            or result.motion_results[1] > threshold
            or result.motion_results[2] > threshold
            or result.motion_results[3] > threshold
            or result.motion_results[4] > threshold
            or result.motion_results[5] > threshold)


test_motion = thresholdTesterFactory(motionCondition)


# %%
plotThresholds(test_motion, 4900, 5100, 10000, title="Motion thresholds")


# %%
motion_threshold = 0.50
print(f"Threshold: {motion_threshold}")
print("==========")
confusion_matrix = test_motion(results, motion_threshold)
utils.printStats(confusion_matrix)


# %% [md]
# ## Shape (Cosine Similarity)


# %%
def shapeCosineCondition(result, threshold):
    return (result.shape_results[0] < threshold
            or result.shape_results[1] < threshold)


test_shape = thresholdTesterFactory(shapeCosineCondition)


# %%
plotThresholds(test_shape, 800, 1000, title="Shape thresholds")


# %%
shape_threshold = 0.92
print(f"Threshold: {shape_threshold}")
print("==========")
confusion_matrix = test_shape(results, shape_threshold)
utils.printStats(confusion_matrix)


# %% [md]
# ## Face (Cosine Similarity)


# %%
def faceCosineCondition(result, threshold):
    return (result.face_result < threshold
            or result.face_result < threshold)


test_face = thresholdTesterFactory(faceCosineCondition)


# %%
plotThresholds(test_face, 9900, 10000, 10000, title="Face thresholds")


# %%
face_threshold = 0.995
print(f"Threshold: {face_threshold}")
print("==========")
confusion_matrix = test_face(results, face_threshold)
utils.printStats(confusion_matrix)


# %% [md]
# # Overall (Without Face)


# %%
def overallCondition(result, _):
    return (result.location_results[0] < location_threshold
            or result.location_results[1] < location_threshold
            or result.motion_results[0] > motion_threshold
            or result.motion_results[1] > motion_threshold
            or result.motion_results[2] > motion_threshold
            or result.motion_results[3] > motion_threshold
            or result.motion_results[4] > motion_threshold
            or result.motion_results[5] > motion_threshold
            or result.shape_results[0] < shape_threshold
            or result.shape_results[1] < shape_threshold)


test_overall = thresholdTesterFactory(overallCondition)

confusion_matrix = test_overall(results, 0.0)
utils.printStats(confusion_matrix)

# %% [md]
# ## Assessing the feedback (without Face)

# %%
differences = utils.importGestureDifferences("test.csv", 5)
feedback_confusion_matrix = [[0] * 4 for _ in range(4)]

def overallConditionWithFeedback(result: Result, _):
    global feedback_confusion_matrix

    predicted_class = 0
    if (result.location_results[0] < location_threshold
            or result.location_results[1] < location_threshold):
        predicted_class = 1
    elif (result.motion_results[0] > motion_threshold
            or result.motion_results[1] > motion_threshold
            or result.motion_results[2] > motion_threshold
            or result.motion_results[3] > motion_threshold
            or result.motion_results[4] > motion_threshold
            or result.motion_results[5] > motion_threshold):
        predicted_class = 2
    elif (result.shape_results[0] < shape_threshold
            or result.shape_results[1] < shape_threshold):
        predicted_class = 3

    actual_class = 0

    g1 = result.gesture1
    g2 = result.gesture2

    if (not differences[g1][g2]["location"]):
        actual_class = 1
    elif (not differences[g1][g2]["motion"]):
        actual_class = 2
    elif (not differences[g1][g2]["shape"]):
        actual_class = 3

    # if (g1 != g2):
    feedback_confusion_matrix[actual_class][predicted_class] += 1
    return predicted_class != 0

test_overall_with_feedback = thresholdTesterFactory(overallConditionWithFeedback)

test_overall_with_feedback(feedback_results, 0.0)

feedback_precisions = [
    feedback_confusion_matrix[i][i]
            / sum([feedback_confusion_matrix[j][i] for j in range(4)])
        if sum([feedback_confusion_matrix[j][i] for j in range(4)]) != 0
        else 0 for i in range(4)
]

feedback_recalls = [
    feedback_confusion_matrix[i][i]
            / sum([feedback_confusion_matrix[i][j] for j in range(4)])
        if sum([feedback_confusion_matrix[i][j] for j in range(4)]) != 0
        else 0 for i in range(4)
]

feedback_f1scores = [
    ((2 * feedback_precisions[i] * feedback_recalls[i])
        / (feedback_precisions[i] + feedback_recalls[i]))
        if (feedback_precisions[i] + feedback_recalls[i]) != 0
        else 0
    for i in range(4)
]

print(f"Precision (No Feedback): {feedback_precisions[0]}")
print(f"Precision (Location)   : {feedback_precisions[1]}")
print(f"Precision (Motion)     : {feedback_precisions[2]}")
print(f"Precision (Shape)      : {feedback_precisions[3]}")
print()
print(f"Recall (No Feedback): {feedback_recalls[0]}")
print(f"Recall (Location)   : {feedback_recalls[1]}")
print(f"Recall (Motion)     : {feedback_recalls[2]}")
print(f"Recall (Shape)      : {feedback_recalls[3]}")
print()
print(f"F1 Score (No Feedback): {feedback_f1scores[0]}")
print(f"F1 Score (Location)   : {feedback_f1scores[1]}")
print(f"F1 Score (Motion)     : {feedback_f1scores[2]}")
print(f"F1 Score (Shape)      : {feedback_f1scores[3]}")
print()
print(f"Accuracy          : {
    sum([feedback_confusion_matrix[i][i] for i in range(4)])
    / len(feedback_results)
}")
print(f"Balanced Accuracy : { sum(feedback_recalls) / len(feedback_recalls) }")
print(f"Balanced Precision: { sum(feedback_precisions) / len(feedback_precisions) }")
print(f"Balanced F1 Score : { sum(feedback_f1scores) / len(feedback_f1scores) }")

feedback_confusion_matrix

# %% [md]
# # Demo (Without Face)


# %%
def testResult(result):
    if (result.location_results[0] < location_threshold
            or result.location_results[1] < location_threshold):
        print("Prediction: Incorrect location")
        return

    if (result.motion_results[0] > motion_threshold
            or result.motion_results[1] > motion_threshold
            or result.motion_results[2] > motion_threshold
            or result.motion_results[3] > motion_threshold
            or result.motion_results[4] > motion_threshold
            or result.motion_results[5] > motion_threshold):
        print("Prediction: Incorrect motion")
        return

    if (result.shape_results[0] < shape_threshold
            or result.shape_results[1] < shape_threshold):
        print("Prediction: Incorrect shape")
        return

    print("Prediction: Correct gesture")


# %%
result = results[random.randrange(0, len(results))]
testResult(result)
print(f"Actual    : {'Correct' if result.gesture1 == result.gesture2 else 'Incorrect'}")


# %% [md]
# # Overall (With Face)


# %%
def overallConditionWithFace(result: Result, _):
    return (result.location_results[0] < location_threshold
            or result.location_results[1] < location_threshold
            or result.motion_results[0] > motion_threshold
            or result.motion_results[1] > motion_threshold
            or result.motion_results[2] > motion_threshold
            or result.motion_results[3] > motion_threshold
            or result.motion_results[4] > motion_threshold
            or result.motion_results[5] > motion_threshold
            or result.shape_results[0] < shape_threshold
            or result.shape_results[1] < shape_threshold
            or result.face_result < face_threshold)


test_overall = thresholdTesterFactory(overallConditionWithFace)

confusion_matrix = test_overall(results, 0.0)
utils.printStats(confusion_matrix)

# %% [md]
# ## Assessing the feedback (with Face)

# %%
differences = utils.importGestureDifferences("test.csv", 5)
feedback_confusion_matrix = [[0] * 5 for _ in range(5)]

def overallConditionWithFaceWithFeedback(result: Result, _):
    global feedback_confusion_matrix

    predicted_class = 0
    if (result.location_results[0] < location_threshold
            or result.location_results[1] < location_threshold):
        predicted_class = 1
    elif (result.motion_results[0] > motion_threshold
            or result.motion_results[1] > motion_threshold
            or result.motion_results[2] > motion_threshold
            or result.motion_results[3] > motion_threshold
            or result.motion_results[4] > motion_threshold
            or result.motion_results[5] > motion_threshold):
        predicted_class = 2
    elif (result.shape_results[0] < shape_threshold
            or result.shape_results[1] < shape_threshold):
        predicted_class = 3
    elif (result.face_result < face_threshold):
        predicted_class = 4

    actual_class = 0

    g1 = result.gesture1
    g2 = result.gesture2

    if (not differences[g1][g2]["location"]):
        actual_class = 1
    elif (not differences[g1][g2]["motion"]):
        actual_class = 2
    elif (not differences[g1][g2]["shape"]):
        actual_class = 3
    elif (not differences[g1][g2]["face"]):
        actual_class = 4

    # if (g1 != g2):
    feedback_confusion_matrix[actual_class][predicted_class] += 1
    return predicted_class != 0

test_overall_with_feedback = thresholdTesterFactory(overallConditionWithFaceWithFeedback)
test_overall_with_feedback(feedback_results, 0.0)

feedback_precisions = [
    feedback_confusion_matrix[i][i]
            / sum([feedback_confusion_matrix[j][i] for j in range(5)])
        if sum([feedback_confusion_matrix[j][i] for j in range(5)]) != 0
        else 0 for i in range(5)
]

feedback_recalls = [
    feedback_confusion_matrix[i][i]
            / sum([feedback_confusion_matrix[i][j] for j in range(5)])
        if sum([feedback_confusion_matrix[i][j] for j in range(5)]) != 0
        else 0 for i in range(5)
]

feedback_f1scores = [
    ((2 * feedback_precisions[i] * feedback_recalls[i])
        / (feedback_precisions[i] + feedback_recalls[i]))
        if (feedback_precisions[i] + feedback_recalls[i]) != 0
        else 0
    for i in range(5)
]

print(f"Precision (No Feedback): {feedback_precisions[0]}")
print(f"Precision (Location)   : {feedback_precisions[1]}")
print(f"Precision (Motion)     : {feedback_precisions[2]}")
print(f"Precision (Shape)      : {feedback_precisions[3]}")
print(f"Precision (Face)       : {feedback_precisions[4]}")
print()
print(f"Recall (No Feedback): {feedback_recalls[0]}")
print(f"Recall (Location)   : {feedback_recalls[1]}")
print(f"Recall (Motion)     : {feedback_recalls[2]}")
print(f"Recall (Shape)      : {feedback_recalls[3]}")
print(f"Recall (Face)       : {feedback_recalls[4]}")
print()
print(f"F1 Score (No Feedback): {feedback_f1scores[0]}")
print(f"F1 Score (Location)   : {feedback_f1scores[1]}")
print(f"F1 Score (Motion)     : {feedback_f1scores[2]}")
print(f"F1 Score (Shape)      : {feedback_f1scores[3]}")
print(f"F1 Score (Face)       : {feedback_f1scores[4]}")
print()
print(f"Accuracy          : {
    sum([feedback_confusion_matrix[i][i] for i in range(5)])
    / len(feedback_results)
}")
print(f"Balanced Accuracy : { sum(feedback_recalls) / len(feedback_recalls) }")
print(f"Balanced Precision: { sum(feedback_precisions) / len(feedback_precisions) }")
print(f"Balanced F1 Score : { sum(feedback_f1scores) / len(feedback_f1scores) }")

feedback_confusion_matrix

# %% [md]
# # Demo (With Face)


# %%
def testResultWithFace(result):
    if (result.location_results[0] < location_threshold
            or result.location_results[1] < location_threshold):
        print("Prediction: Incorrect location")
        return

    if (result.motion_results[0] > motion_threshold
            or result.motion_results[1] > motion_threshold
            or result.motion_results[2] > motion_threshold
            or result.motion_results[3] > motion_threshold
            or result.motion_results[4] > motion_threshold
            or result.motion_results[5] > motion_threshold):
        print("Prediction: Incorrect motion")
        return

    if (result.shape_results[0] < shape_threshold
            or result.shape_results[1] < shape_threshold):
        print("Prediction: Incorrect shape")
        return

    if (result.face_result < face_threshold):
        print("Prediction: Incorrect face")
        return

    print("Prediction: Correct gesture")


# %%
result = results[random.randrange(0, len(results))]
testResultWithFace(result)
print(f"Actual    : {'Correct' if result.gesture1 == result.gesture2 else 'Incorrect'}")
