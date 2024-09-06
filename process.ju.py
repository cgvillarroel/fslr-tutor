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

def processRange(start, stop):
    for i in range(start, stop):
        print(f"gesture {i} ", end="\r")
        results = []
        clip1 = gestures[i].clips[0]

        # correct results (same gesture, different clip)
        for j in range(1, 16):
            result = Result(i, i)
            result.location_results = location.compareHandLocations(clip1, gestures[i].clips[j])
            result.motion_results = motion.compareMotions(clip1, gestures[i].clips[j])
            result.shape_results = shape.compareHandShapes(clip1, gestures[i].clips[j])
            results.append(result)

        # incorrect results (different gestures)
        for j in range(0, i):
            result = Result(i, j)
            result.location_results = location.compareHandLocations(clip1, gestures[j].clips[0])
            result.motion_results = motion.compareMotions(clip1, gestures[j].clips[0])
            result.shape_results = shape.compareHandShapes(clip1, gestures[j].clips[0])
            results.append(result)

        # incorrect results (different gestures, part 2)
        for j in range(i + 1, 16):
            result = Result(i, j)
            result.location_results = location.compareHandLocations(clip1, gestures[j].clips[0])
            result.motion_results = motion.compareMotions(clip1, gestures[j].clips[0])
            result.shape_results = shape.compareHandShapes(clip1, gestures[j].clips[0])
            results.append(result)

        with open(f"results/{i}.pkl", "wb") as writer:
            pickle.dump(results, writer)


# %%
# multithreading
import threading

threads = []
for idx in range(4):
    threads.append(threading.Thread(target=processRange, args=(idx * 4, idx * 4 + 4)))

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()
