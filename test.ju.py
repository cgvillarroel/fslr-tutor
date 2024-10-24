# %% [md]
# ## Import modules
# %%
import pickle
import random
import structs.functions as utils
from structs.types import Result

# %% [md]
# ## Import comparison results and split
# ### Binary
# %%
# import
results_binary: list[Result] = []
for i in range(16):
    with open(f"results/binary/standard_balanced-cosine/{i}.pkl", "rb") as reader:
        results_binary.extend(pickle.load(reader))

random.shuffle(results_binary)
y_binary = [1 if (r.gesture1 == r.gesture2) else 0 for r in results_binary]

# train/test split
x_train_binary, x_test_binary, y_train_binary, y_test_binary = utils.train_test_split(results_binary, y_binary, 0.75)

# cross-validation folds
x_train_folds, x_test_folds, y_train_folds, y_test_folds = utils.train_test_folds(x_train_binary, y_train_binary, 12)

# all counts
print(f"Dataset: {len(results_binary)}")
print(f"  Train Split: {len(x_train_binary)}")
for idx, fold in enumerate(x_train_folds):
    print(f"    Train Fold {idx}: {len(fold)}")
for idx, fold in enumerate(x_test_folds):
    print(f"    Test Fold {idx}: {len(fold)}")
print(f"  Test Split: {len(x_test_binary)}")

# %% [md]
# ### Multiclass
# %%
differences = utils.importGestureDifferences("test.csv", 5)

x_multiclass_diff = [x for x in results_binary if (x.gesture1 < 5 and x.gesture2 < 5) and (x.gesture1 != x.gesture2)]
x_multiclass_same = [x for x in results_binary if (x.gesture1 < 5 and x.gesture2 < 5) and (x.gesture1 == x.gesture2)]

# oversample missing stuff
x_multiclass = x_multiclass_same
for _ in range(len(x_multiclass_same) * 4):
    sample = x_multiclass_diff[random.randint(0, len(x_multiclass_diff) - 1)]
    x_multiclass.append(sample)

random.shuffle(x_multiclass)
y_multiclass = [differences[x.gesture1][x.gesture2] for x in x_multiclass]

_, x_test_multiclass, _, y_test_multiclass = utils.train_test_split(x_multiclass, y_multiclass, 0.75)

# count
print(f"Dataset: {len(x_test_multiclass)}")
print(f"  Test Split: {len(y_test_multiclass)}")

# %% [md]
# ## Finding thresholds
# ### Location
# %%
# graph different training thresholds
utils.plot_thresholds(
    threshold_name="location",
    title="Thresholds for location module only",
    iterator=range(900, 1000),
    test_function=utils.test_location,
    x_values=x_train_folds[0],
    y_values=y_train_folds[0])

# %%
# test set threshold
utils.test_thresholds_binary(
    title="Location module only",
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
    title="Shoulder thresholds for motion module Only",
    iterator=range(500),
    scale=1,
    test_function=utils.test_motion,
    x_values=x_train_folds[1],
    y_values=y_train_folds[1],
    other_thresholds={
        "motion_elbow": 1,
        "motion_wrist": 1,
    })

# %%
# test set threshold
utils.test_thresholds_binary(
    title="Motion module only",
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
    title="Elbow thresholds for motion module only",
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
utils.test_thresholds_binary(
    title="Motion module only",
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
    title="Wrist thresholds for motion module only",
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
utils.test_thresholds_binary(
    title="Motion module only",
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
    title="Threholds for shape module only",
    iterator=range(1000),
    test_function=utils.test_shape,
    x_values=x_train_folds[4],
    y_values=y_train_folds[4])

# %%
# test set threshold
utils.test_thresholds_binary(
    title="Shape module only",
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
    title="Thresholds for face module only",
    iterator=range(1000),
    test_function=utils.test_face,
    x_values=x_train_folds[5],
    y_values=y_train_folds[5])

# %%
# test set threshold
utils.test_thresholds_binary(
    title="Face module only",
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
    title="Location thresholds with all modules",
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
utils.test_thresholds_binary(
    title="All modules combined",
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
    title="Shoulder thresholds with all modules",
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
utils.test_thresholds_binary(
    title="All modules combined",
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
    title="Elbow thresholds with all modules",
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
utils.test_thresholds_binary(
    title="All modules combined",
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
    title="Wrist thresholds with all modules",
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
utils.test_thresholds_binary(
    title="All modules combined",
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
    title="Shape thresholds with all modules",
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
utils.test_thresholds_binary(
    title="All modules combined",
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
    title="Face thresholds with all modules",
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
utils.test_thresholds_binary(
    title="All modules combined",
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
# ## Overall Binary Testing
# %%
thresholds = {
    "location": 0.975,
    "motion_shoulder": 0.5,
    "motion_elbow": 0.5,
    "motion_wrist": 0.5,
    "shape": 0.92,
    "face": 0.995,
}

utils.test_thresholds_binary(
    title="Final metrics with face module (Binary Classification)",
    x_values=x_test_binary,
    y_values=y_test_binary,
    thresholds=thresholds,
    test_function=utils.test_with_face_binary)

# %% [md]
# ## Overall Binary Testing without Face
# %%
thresholds = {
    "location": 0.975,
    "motion_shoulder": 0.5,
    "motion_elbow": 0.5,
    "motion_wrist": 0.5,
    "shape": 0.92,
}

utils.test_thresholds_binary(
    title="Final metrics without face module (Binary Classification)",
    x_values=x_test_binary,
    y_values=y_test_binary,
    thresholds=thresholds,
    test_function=utils.test_without_face_binary)

# %% [md]
# ## Overall Multiclass Testing
# %%
thresholds = {
    "location": 0.975,
    "motion_shoulder": 0.5,
    "motion_elbow": 0.5,
    "motion_wrist": 0.5,
    "shape": 0.92,
    "face": 0.995,
}

utils.test_thresholds_multiclass(
    title="Final metrics with face module (Multiclass Classification)",
    x_values=x_test_multiclass,
    y_values=y_test_multiclass,
    thresholds=thresholds,
    test_function=utils.test_with_face_multiclass)

# %% [md]
# ## Overall Multiclass Testing without Face
# %%
thresholds = {
    "location": 0.975,
    "motion_shoulder": 0.5,
    "motion_elbow": 0.5,
    "motion_wrist": 0.5,
    "shape": 0.92,
}

utils.test_thresholds_multiclass(
    title="Final metrics without face module (Multiclass Classification)",
    x_values=x_test_multiclass,
    y_values=y_test_multiclass,
    thresholds=thresholds,
    test_function=utils.test_without_face_multiclass)
