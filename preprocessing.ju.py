# %%
import cv2
import mediapipe as mp
from structs.types import NormalizedLandmark, Frame, Clip, Gesture

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# %%
# util: draw landmarks
draw_landmarks = False


def drawLandmarks(image, results):
    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow("landmarks", cv2.flip(image, 1))


# %%
# util: extract landmarks
# no longer needed, rescaled with ffmpeg
# downsample_width = 320
# downsample_height = 240


def extractLandmarks(file_name):
    cap = cv2.VideoCapture(file_name)
    results = []
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0) as holistic:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # cv2.resize(frame, (downsample_width, downsample_height),
            #            interpolation=cv2.INTER_LINEAR)

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(frame)
            results.append(result)

            if draw_landmarks:
                drawLandmarks(frame, result)

            if cv2.waitKey(5) == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()
    return results

# %%
# util: normalize


def normalizeResults(results):
    normalized_results = Clip()
    for result in results:
        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")

        landmarks = result.pose_landmarks.landmark

        for landmark in landmarks:
            min_x = landmark.x if landmark.x < min_x else min_x
            min_y = landmark.y if landmark.y < min_y else min_y
            max_x = landmark.x if landmark.x > max_x else max_x
            max_y = landmark.y if landmark.y > max_y else max_y

        normalized_landmarks = Frame()

        if result.pose_landmarks is not None:
            for landmark in result.pose_landmarks.landmark:
                normalized_x = (landmark.x - min_x) / (max_x - min_x)
                normalized_y = (landmark.y - min_y) / (max_y - min_y)
                normalized_landmarks.pose_landmarks.append(NormalizedLandmark(normalized_x, normalized_y, landmark.visibility))

        if result.face_landmarks is not None:
            for landmark in result.face_landmarks.landmark:
                normalized_x = (landmark.x - min_x) / (max_x - min_x)
                normalized_y = (landmark.y - min_y) / (max_y - min_y)
                normalized_landmarks.face_landmarks.append(NormalizedLandmark(normalized_x, normalized_y, landmark.visibility))

        if result.left_hand_landmarks is not None:
            for landmark in result.left_hand_landmarks.landmark:
                normalized_x = (landmark.x - min_x) / (max_x - min_x)
                normalized_y = (landmark.y - min_y) / (max_y - min_y)
                normalized_landmarks.left_hand_landmarks.append(NormalizedLandmark(normalized_x, normalized_y, landmark.visibility))

        if result.right_hand_landmarks is not None:
            for landmark in result.right_hand_landmarks.landmark:
                normalized_x = (landmark.x - min_x) / (max_x - min_x)
                normalized_y = (landmark.y - min_y) / (max_y - min_y)
                normalized_landmarks.right_hand_landmarks.append(NormalizedLandmark(normalized_x, normalized_y, landmark.visibility))

        normalized_results.frames.append(normalized_landmarks)

    return normalized_results


# %%
# actual preprocessing
# save to a binary in chunks to avoid repeating
import os
import pickle


dataset_path = "dataset"
clips_path = f"{dataset_path}/scaled_clips"
chunks_path = f"{dataset_path}/chunks"

def preprocessRange(start, stop):
    for gesture_idx in range(start, stop):
        gesture = Gesture()
        clip_count = len(next(os.walk(f"{clips_path}/{gesture_idx}"))[2])
        for clip_idx in range(clip_count):
            print(f"Processing gesture {gesture_idx}, clip {clip_idx}    ", end="\r")
            result = extractLandmarks(f"{clips_path}/{gesture_idx}/{clip_idx}.MOV")
            normalized_result = normalizeResults(result)
            gesture.clips.append(normalized_result)

        # redundancy in case of failure
        with open(f"{chunks_path}/{gesture_idx}.pkl", "wb") as chunk_writer:
            pickle.dump(gesture, chunk_writer)

# %%
import threading

threads = []
for idx in range(21):
    threads.append(threading.Thread(target=preprocessRange, args=(idx*5, idx*5 + 5)))

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

# %%
# sample reading
with open(f"{dataset_path}/preprocessed.pkl", "rb") as reader:
    loaded_gesture: Gesture = pickle.load(reader)
    for landmark in loaded_gesture.clips[0].frames[0].pose_landmarks:
        print(f"norm x: {landmark.x:.5f}, norm y: {landmark.y:.5}, visibility: {landmark.visibility:.5f}")
