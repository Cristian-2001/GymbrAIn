import numpy as np
import cv2
import torch
from ultralytics import YOLO
from tensorflow.keras.models import load_model

model_YOLO = YOLO("yolov8m-pose.pt")  # load an official model


def estimate_poses(video_path, num_frames=30):
    """Estimate poses in a video and return the results. The number of frames to process can be specified"""
    # Perform pose estimation
    poses = []
    poses_norm = []
    video = cv2.VideoCapture(video_path)
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frames // num_frames)  # Step for frame sampling (process at most num_frames frames)
    counter = 0

    for i in range(0, frames, step):
        if counter >= num_frames:
            break
        counter += 1

        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        if not ret:
            break

        # frame = cv2.resize(frame, (416, 416))   # Resize for YOLO

        results = model_YOLO(frame)

        # Append the poses to the results list
        if not results[0]:
            poses.append(torch.zeros((1, 17, 2), dtype=torch.float32))
        else:
            poses_norm.append(results[0][0].keypoints.xyn)  # Original line: it returns the xyn coordinates
            poses.append(results[0][0].keypoints.xy)  # Modified to return only the xy coordinates

    video.release()
    return poses, poses_norm


def classify_exercise(poses):
    print("CLASSIFY_EXERCISE")
    model_att = load_model("model_att.keras")

    poses = np.array(poses)
    poses = poses.flatten()
    poses = poses.reshape(1, 1020)
    prediction = model_att.predict(poses)

    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class
