# %%
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
import pickle

data = []

with open("dataset/gestures/0.pkl", "rb") as reader:
    data.append(pickle.load(reader))

with open("dataset/gestures/6.pkl", "rb") as reader:
    data.append(pickle.load(reader))

print(location.compareHandLocations(data[0].clips[0], data[0].clips[1]))
print(location.compareHandLocations(data[0].clips[0], data[1].clips[0]))

print(motion.compareMotions(data[0].clips[0], data[0].clips[1]))
print(motion.compareMotions(data[0].clips[0], data[1].clips[0]))

print(shape.compareHandShapes(data[0].clips[0], data[0].clips[1]))
print(shape.compareHandShapes(data[0].clips[0], data[1].clips[0]))
