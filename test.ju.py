# %%
# import data
import pickle
import random
from structs.types import Result

results: list[Result] = []

for i in range(16):
    with open(f"results/{i}.pkl", "rb") as reader:
        results.extend(pickle.load(reader))

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


# %% [md]
# ## Shape (Cosine Similarity)


# %%
def shapeCosineCondition(result, threshold):
    return (result.shape_results[0] < threshold
            or result.shape_results[1] < threshold)

test_shape = thresholdTesterFactory(shapeCosineCondition)


# %%
plotThresholds(test_shape, 800, 1000, title="Shape thresholds")


# %% [md]
# ## Face (Cosine Similarity)


# %%
def faceCosineCondition(result, threshold):
    return (result.face_result < threshold
            or result.face_result < threshold)

test_face = thresholdTesterFactory(faceCosineCondition)


# %%
plotThresholds(test_face, 9900, 10000, 10000, title="Face thresholds")


# %% [md]
# # Overall (Without Face)


# %%
location_threshold = 0.975
motion_threshold = 0.50
shape_threshold = 0.92

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

    print("Correct gesture")


# %%
result = results[random.randrange(0, len(results))]
testResult(result)
print(f"Actual    : {'Correct' if result.gesture1 == result.gesture2 else 'Incorrect'}")


# %% [md]
# # Overall (With Face)


# %%
location_threshold = 0.975
motion_threshold = 0.50
shape_threshold = 0.92
face_threshold = 0.998

def overallConditionWithFace(result, _):
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

    print("Correct gesture")


# %%
result = results[random.randrange(0, len(results))]
testResultWithFace(result)
print(f"Actual    : {'Correct' if result.gesture1 == result.gesture2 else 'Incorrect'}")
