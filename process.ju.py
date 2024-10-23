# %%
# dots in file names mess up imports, manually do it
import importlib.util
from structs.types import Result


def import_from_file(file_name, module_name):

    spec = importlib.util.spec_from_file_location(
        name=module_name,
        location=file_name,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


location = import_from_file("modules/location.ju.py", "location")
motion = import_from_file("modules/motion.ju.py", "motion")
shape = import_from_file("modules/shape.ju.py", "shape")
face = import_from_file("modules/face.ju.py", "shape")


# %%
# load data
import pickle

gestures = []

for i in range(16):
    print(f"gesture {i}  ", end="\r")
    with open(f"dataset/chunks/{i}.pkl", "rb") as reader:
        gestures.append(pickle.load(reader))


# %%
# run through modules
from random import randint


def processBinary(
        output_folder="results",
        location_module=location.compareHandLocations,
        motion_module=motion.compareMotions,
        shape_module=shape.compareHandShapesCosine,
        face_module=face.compareFacesCosine):
    for i in range(16):
        print(f"gesture {i} ", end="\r")
        results = []
        # choose random "reference" clip
        clip1_idx = randint(0, 15)
        clip1 = gestures[i].clips[clip1_idx]

        # correct results (same gesture, different clip)
        for _ in range(0, 15):
            try:
                # get random other "learner" clip of the same gesture
                random_idx = randint(0, 14)
                clip2_idx = random_idx if random_idx < clip1_idx else random_idx + 1
                clip2 = gestures[i].clips[clip2_idx]

                # run through modules
                result = Result(i, i)
                result.location_results = location_module(clip1, clip2)
                result.motion_results = motion_module(clip1.frames, clip2.frames)
                result.shape_results = shape_module(clip1, clip2)
                result.face_result = face_module(clip1, clip2)

                results.append(result)

            except:
                pass

        # incorrect results (different gestures)
        for _ in range(0, 15):
            try:
                # some don't have facial features, just skip them
                # get random other "learner" clip of a different gesture
                random_idx = randint(0, 14)
                gesture2_idx = random_idx if random_idx < i else random_idx + 1
                clip2 = gestures[gesture2_idx].clips[randint(0, 15)]

                # run through modules
                result = Result(i, gesture2_idx)
                result.location_results = location_module(clip1, clip2)
                result.motion_results = motion_module(clip1.frames, clip2.frames)
                result.shape_results = shape_module(clip1, clip2)
                result.face_result = face_module(clip1, clip2)

                results.append(result)

            except:
                pass

        # save after every gesture (in case of fault)
        with open(f"{output_folder}/{i}.pkl", "wb") as writer:
            pickle.dump(results, writer)

# %%
# for feedback assessment


def processMultiClass(
        output_folder="results/feedback",
        location_module=location.compareHandLocations,
        motion_module=motion.compareMotions,
        shape_module=shape.compareHandShapesCosine,
        face_module=face.compareFacesCosine):
    for i in range(5):
        print(f"gesture {i} ", end="\r")
        results = []

        # choose random "reference" clip
        clip1_idx = randint(0, 5)
        clip1 = gestures[i].clips[clip1_idx]

        for j in range(5):
            for _ in range(1, 16):
                try:
                    clip2 = None

                    gesture2_idx = i
                    result = None
                    # correct results (same gesture, different clip)
                    if (i == j):
                        # get random other "learner" clip of the same gesture
                        random_idx = randint(0, 3)
                        clip2_idx = random_idx if random_idx < clip1_idx else random_idx + 1
                        clip2 = gestures[i].clips[clip2_idx]

                    # incorrect results (different gestures)
                    else:
                        # get random other "learner" clip of a different gesture
                        random_idx = randint(0, 3)
                        gesture2_idx = random_idx if random_idx < i else random_idx + 1
                        clip2 = gestures[gesture2_idx].clips[randint(0, 4)]

                    # run through modules
                    result = Result(i, gesture2_idx)
                    result.location_results = location_module(clip1, clip2)
                    result.motion_results = motion_module(clip1.frames, clip2.frames)
                    result.shape_results = shape_module(clip1, clip2)
                    result.face_result = face_module(clip1, clip2)

                    results.append(result)

                except:
                    pass

        # save after every gesture (in case of fault)
        with open(f"{output_folder}/{i}.pkl", "wb") as writer:
            pickle.dump(results, writer)


# %%

binary_configs = [
    {
        "output_folder": "results/binary/naive-cosine",
        "motion_module": motion.compareMotions,
        "shape_module": shape.compareHandShapesCosine,
        "face_module": face.compareFacesCosine,
    },
    {
        "output_folder": "results/binary/segmented-cosine",
        "motion_module": motion.compareMotionsSegmented,
        "shape_module": shape.compareHandShapesCosine,
        "face_module": face.compareFacesCosine,
    },
    {
        "output_folder": "results/binary/naive-euclid",
        "motion_module": motion.compareMotions,
        "shape_module": shape.compareHandShapesEuclid,
        "face_module": face.compareFacesEuclid,
    },
    {
        "output_folder": "results/binary/segmented-euclid",
        "motion_module": motion.compareMotionsSegmented,
        "shape_module": shape.compareHandShapesEuclid,
        "face_module": face.compareFacesEuclid,
    },
]

for config in binary_configs:
    print(config["output_folder"])
    processBinary(
        output_folder=config["output_folder"],
        motion_module=config["motion_module"],
        shape_module=config["shape_module"],
        face_module=config["face_module"])


# %%
multiclass_configs = [
    {
        "output_folder": "results/multiclass/naive-cosine",
        "motion_module": motion.compareMotions,
        "shape_module": shape.compareHandShapesCosine,
        "face_module": face.compareFacesCosine,
    },
    {
        "output_folder": "results/multiclass/segmented-cosine",
        "motion_module": motion.compareMotionsSegmented,
        "shape_module": shape.compareHandShapesCosine,
        "face_module": face.compareFacesCosine,
    },
    {
        "output_folder": "results/multiclass/naive-euclid",
        "motion_module": motion.compareMotions,
        "shape_module": shape.compareHandShapesEuclid,
        "face_module": face.compareFacesEuclid,
    },
    {
        "output_folder": "results/multiclass/segmented-euclid",
        "motion_module": motion.compareMotionsSegmented,
        "shape_module": shape.compareHandShapesEuclid,
        "face_module": face.compareFacesEuclid,
    },
]

for config in multiclass_configs:
    print(config["output_folder"])
    processMultiClass(
        output_folder=config["output_folder"],
        motion_module=config["motion_module"],
        shape_module=config["shape_module"],
        face_module=config["face_module"])
