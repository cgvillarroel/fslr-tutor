# %%
# dots in file names mess up imports, manually do it
import importlib.util


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
    with open(f"dataset/gestures/{i}.pkl", "rb") as reader:
        gestures.append(pickle.load(reader))


# %%
# run through modules
from structs.types import Result

super_sample_scale = 1

def processRange(start, stop):
    for i in range(start, stop):
        print(f"gesture {i} ", end="\r")
        results = []
        clip1 = gestures[i].clips[0]

        # correct results (same gesture, different clip)
        for j in range(1, 16):
            result = Result(i, i)
            clip2 = gestures[i].clips[j]
            result.location_results = location.compareHandLocations(clip1, clip2)
            result.motion_results = motion.compareMotions(clip1, clip2)
            result.shape_results = shape.compareHandShapesCosine(clip1, clip2)
            result.face_result = face.compareFacesCosine(clip1, clip2)
            for k in range(super_sample_scale):
                results.append(result)

        # incorrect results (different gestures)
        for j in range(0, i):
            result = Result(i, j)
            for k in range(super_sample_scale):
                clip2 = gestures[j].clips[k]
                result.location_results = location.compareHandLocations(clip1, clip2)
                result.motion_results = motion.compareMotions(clip1, clip2)
                result.shape_results = shape.compareHandShapesCosine(clip1, clip2)
                result.face_result = face.compareFacesCosine(clip1, clip2)
                results.append(result)

        # incorrect results (different gestures, part 2)
        for j in range(i + 1, 16):
            result = Result(i, j)
            for k in range(super_sample_scale):
                clip2 = gestures[j].clips[k]
                result.location_results = location.compareHandLocations(clip1, clip2)
                result.motion_results = motion.compareMotions(clip1, clip2)
                result.shape_results = shape.compareHandShapesCosine(clip1, clip2)
                result.face_result = face.compareFacesCosine(clip1, clip2)
                results.append(result)

        with open(f"results/{i}.pkl", "wb") as writer:
            pickle.dump(results, writer)


# %%
processRange(0, 16)


# %%
# multithreading
# import threading
#
# threads = []
# for idx in range(8):
#     threads.append(threading.Thread(target=processRange, args=(idx * 2, idx * 2 + 2)))
#
# for thread in threads:
#     thread.start()
#
# for thread in threads:
#     thread.join()
