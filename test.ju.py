# %% [md]
# ## Import modules
# %%
import pickle
import random
import structs.functions as utils
from structs.types import Result

# %% [md]
# ## Import comparison results and split
# %%
# import
results: list[Result] = []
for i in range(16):
    with open(f"results/binary/standard_balanced-cosine/{i}.pkl", "rb") as reader:
        results.extend(pickle.load(reader))

random.shuffle(results)

y = [1 if (r.gesture1 == r.gesture2) else 0 for r in results]

# train/test split
x_train, x_test, y_train, y_test = utils.train_test_split(results, y, 0.75)

# cross-validation folds
x_train_folds, x_test_folds, y_train_folds, y_test_folds = utils.train_test_folds(x_train, y_train, 12)

# %%
# all counts
print(f"Dataset: {len(results)}")
print(f"  Train Split: {len(x_train)}")
for idx, fold in enumerate(x_train_folds):
    print(f"    Train Fold {idx}: {len(fold)}")
for idx, fold in enumerate(x_test_folds):
    print(f"    Test Fold {idx}: {len(fold)}")
print(f"  Test Split: {len(x_test)}")

# %% [md]
# ## Finding thresholds
# ### Location
# %%
# graph different training thresholds
utils.plot_thresholds(
    threshold_name="location",
    iterator=range(900, 1000),
    test_function=utils.test_location,
    x_values=x_train_folds[0],
    y_values=y_train_folds[0])

# %%
# test set threshold
utils.test_thresholds(
    x_values=x_test_folds[0],
    y_values=y_test_folds[0],
    thresholds={"location": 0.975},
    test_function=utils.test_location)

# %% [md]
# ### Motion
# #### Shoulder
# %%
# graph different training thresholds
utils.plot_thresholds(
    threshold_name="motion_shoulder",
    iterator=range(4900, 5100),
    scale=1000,
    test_function=utils.test_motion,
    x_values=x_train_folds[1],
    y_values=y_train_folds[1],
    other_thresholds={
        "motion_elbow": 1,
        "motion_wrist": 1,
    })

# %%
# test set threshold
utils.test_thresholds(
    x_values=x_test_folds[1],
    y_values=y_test_folds[1],
    thresholds={
        "motion_shoulder": 0.5,
        "motion_elbow": 1,
        "motion_wrist": 1,
    },
    test_function=utils.test_motion)


# %% [md]
# #### Elbow
# %%
# graph different training thresholds
utils.plot_thresholds(
    threshold_name="motion_elbow",
    iterator=range(4900, 5100),
    scale=1000,
    test_function=utils.test_motion,
    x_values=x_train_folds[2],
    y_values=y_train_folds[2],
    other_thresholds={
        "motion_shoulder": 1,
        "motion_wrist": 1,
    })

# %%
# test set threshold
utils.test_thresholds(
    x_values=x_test_folds[2],
    y_values=y_test_folds[2],
    thresholds={
        "motion_shoulder": 1,
        "motion_elbow": 0.5,
        "motion_wrist": 1,
    },
    test_function=utils.test_motion)


# %% [md]
# #### Wrist
# %%
# graph different training thresholds
utils.plot_thresholds(
    threshold_name="motion_wrist",
    iterator=range(4900, 5100),
    scale=1000,
    test_function=utils.test_motion,
    x_values=x_train_folds[3],
    y_values=y_train_folds[3],
    other_thresholds={
        "motion_shoulder": 1,
        "motion_elbow": 1,
    })

# %%
# test set threshold
utils.test_thresholds(
    x_values=x_test_folds[3],
    y_values=y_test_folds[3],
    thresholds={
        "motion_shoulder": 1,
        "motion_elbow": 1,
        "motion_wrist": 0.5,
    },
    test_function=utils.test_motion)


# %% [md]
# ### Shape
# %%
# graph different training thresholds
utils.plot_thresholds(
    threshold_name="shape",
    iterator=range(1000),
    test_function=utils.test_shape,
    x_values=x_train_folds[4],
    y_values=y_train_folds[4])

# %%
# test set threshold
utils.test_thresholds(
    x_values=x_test_folds[4],
    y_values=y_test_folds[4],
    thresholds={
        "shape": 0.92,
    },
    test_function=utils.test_shape)


# %% [md]
# ### Face
# %%
# graph different training thresholds
utils.plot_thresholds(
    threshold_name="face",
    iterator=range(1000),
    test_function=utils.test_face,
    x_values=x_train_folds[5],
    y_values=y_train_folds[5])

# %%
# test set threshold
utils.test_thresholds(
    x_values=x_test_folds[5],
    y_values=y_test_folds[5],
    thresholds={
        "face": 0.995,
    },
    test_function=utils.test_face)


# %% [md]
# ## Optimizing thresholds
# ### Location
# %%
# graph different training thresholds
utils.plot_thresholds(
    threshold_name="location",
    iterator=range(900, 1000),
    test_function=utils.test_with_face_binary,
    x_values=x_train_folds[6],
    y_values=y_train_folds[6],
    other_thresholds={
        "motion_shoulder": 0.5,
        "motion_elbow": 0.5,
        "motion_wrist": 0.5,
        "shape": 0.92,
        "face": 0.995,
    })

# %%
# test set threshold
utils.test_thresholds(
    x_values=x_test_folds[6],
    y_values=y_test_folds[6],
    thresholds={
        "location": 0.975,
        "motion_shoulder": 0.5,
        "motion_elbow": 0.5,
        "motion_wrist": 0.5,
        "shape": 0.92,
        "face": 0.995,
    },
    test_function=utils.test_with_face_binary)

# %% [md]
# ### Motion
# #### Shoulder
# %%
# graph different training thresholds
utils.plot_thresholds(
    threshold_name="motion_shoulder",
    iterator=range(4900, 5100),
    scale=1000,
    test_function=utils.test_with_face_binary,
    x_values=x_train_folds[7],
    y_values=y_train_folds[7],
    other_thresholds={
        "location": 0.975,
        "motion_elbow": 0.5,
        "motion_wrist": 0.5,
        "shape": 0.92,
        "face": 0.995,
    })

# %%
# test set threshold
utils.test_thresholds(
    x_values=x_test_folds[7],
    y_values=y_test_folds[7],
    thresholds={
        "location": 0.975,
        "motion_shoulder": 0.5,
        "motion_elbow": 0.5,
        "motion_wrist": 0.5,
        "shape": 0.92,
        "face": 0.995,
    },
    test_function=utils.test_with_face_binary)


# %% [md]
# #### Elbow
# %%
# graph different training thresholds
utils.plot_thresholds(
    threshold_name="motion_elbow",
    iterator=range(4900, 5100),
    scale=1000,
    test_function=utils.test_with_face_binary,
    x_values=x_train_folds[8],
    y_values=y_train_folds[8],
    other_thresholds={
        "location": 0.975,
        "motion_shoulder": 0.5,
        "motion_wrist": 0.5,
        "shape": 0.92,
        "face": 0.995,
    })

# %%
# test set threshold
utils.test_thresholds(
    x_values=x_test_folds[8],
    y_values=y_test_folds[8],
    thresholds={
        "location": 0.975,
        "motion_shoulder": 0.5,
        "motion_elbow": 0.5,
        "motion_wrist": 0.5,
        "shape": 0.92,
        "face": 0.995,
    },
    test_function=utils.test_with_face_binary)


# %% [md]
# #### Wrist
# %%
# graph different training thresholds
utils.plot_thresholds(
    threshold_name="motion_wrist",
    iterator=range(4900, 5100),
    scale=1000,
    test_function=utils.test_with_face_binary,
    x_values=x_train_folds[9],
    y_values=y_train_folds[9],
    other_thresholds={
        "location": 0.975,
        "motion_shoulder": 0.5,
        "motion_elbow": 0.5,
        "shape": 0.92,
        "face": 0.995,
    })

# %%
# test set threshold
utils.test_thresholds(
    x_values=x_test_folds[9],
    y_values=y_test_folds[9],
    thresholds={
        "location": 0.975,
        "motion_shoulder": 0.5,
        "motion_elbow": 0.5,
        "motion_wrist": 0.5,
        "shape": 0.92,
        "face": 0.995,
    },
    test_function=utils.test_with_face_binary)


# %% [md]
# ### Shape
# %%
# graph different training thresholds
utils.plot_thresholds(
    threshold_name="shape",
    iterator=range(1000),
    test_function=utils.test_with_face_binary,
    x_values=x_train_folds[10],
    y_values=y_train_folds[10],
    other_thresholds={
        "location": 0.975,
        "motion_shoulder": 0.5,
        "motion_elbow": 0.5,
        "motion_wrist": 0.5,
        "face": 0.995,
    })

# %%
# test set threshold
utils.test_thresholds(
    x_values=x_test_folds[10],
    y_values=y_test_folds[10],
    thresholds={
        "location": 0.975,
        "motion_shoulder": 0.5,
        "motion_elbow": 0.5,
        "motion_wrist": 0.5,
        "shape": 0.92,
        "face": 0.995,
    },
    test_function=utils.test_with_face_binary)


# %% [md]
# ### Face
# %%
# graph different training thresholds
utils.plot_thresholds(
    threshold_name="face",
    iterator=range(1000),
    test_function=utils.test_with_face_binary,
    x_values=x_train_folds[11],
    y_values=y_train_folds[11],
    other_thresholds={
        "location": 0.975,
        "motion_shoulder": 0.5,
        "motion_elbow": 0.5,
        "motion_wrist": 0.5,
        "shape": 0.92,
    })

# %%
# test set threshold
utils.test_thresholds(
    x_values=x_test_folds[11],
    y_values=y_test_folds[11],
    thresholds={
        "location": 0.975,
        "motion_shoulder": 0.5,
        "motion_elbow": 0.5,
        "motion_wrist": 0.5,
        "shape": 0.92,
        "face": 0.995,
    },
    test_function=utils.test_with_face_binary)


# %% [md]
# ## Overall Testing
# %%
thresholds = {
    "location": 0.975,
    "motion_shoulder": 0.5,
    "motion_elbow": 0.5,
    "motion_wrist": 0.5,
    "shape": 0.92,
    "face": 0.995,
}

utils.test_thresholds(
    x_values=x_test,
    y_values=y_test,
    thresholds=thresholds,
    test_function=utils.test_with_face_binary)
