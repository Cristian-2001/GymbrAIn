import numpy as np
import torch
import cv2
import json
import matplotlib.pyplot as plt
from PIL import Image

import geometry


def find_frames(video_path, min_frame):
    print("FIND_FRAMES")
    frame_list = []
    video = cv2.VideoCapture(video_path)
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frames // 30)  # Step for frame sampling (process at most num_frames frames)
    counter = 0

    for i in range(0, frames, step):
        if counter >= 30:
            break

        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame_video = video.read()
        if not ret:
            break

        if counter in min_frame:
            frame_list.append(frame_video)
        counter += 1

    video.release()
    return frame_list


def load_json(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data


def find_same_poses(poses_user, poses_user_norm, poses_luca, poses_luca_norm, ex_name):
    """
    Find the most similar poses between the user and luca, it calculates only the first frame because the other
    should adjust by itself
    :param poses_user: user's poses
    :param poses_user_norm: user's normalized poses
    :param poses_luca: Luca's poses
    :param poses_luca_norm: Luca's normalized poses
    :param ex_name: name of the exercise
    :return: a list of tuples with the most similar poses and a boolean that indicates if the poses have been switched
    """
    print("FIND_SAME_POSES")
    switch = False
    p_user_0 = poses_user_norm[0].reshape(1, -1)
    p_luca_0 = poses_luca_norm[0].reshape(1, -1)
    p_luca_1 = poses_luca_norm[1].reshape(1, -1)
    frame_name_0 = f"videoLuca/frames/{ex_name}_0.jpg"
    frame_name_1 = f"videoLuca/frames/{ex_name}_1.jpg"

    dist_0 = torch.norm(p_user_0 - p_luca_0, p=1).item()
    dist_1 = torch.norm(p_user_0 - p_luca_1, p=1).item()
    print(dist_0, dist_1)
    if dist_1 < dist_0:
        poses_tuples = [(poses_user[0], poses_luca[1], frame_name_1), (poses_user[1], poses_luca[0], frame_name_0)]
        poses_luca_norm[0], poses_luca_norm[1] = poses_luca_norm[1], poses_luca_norm[0]
        switch = True
    else:
        poses_tuples = [(poses_user[0], poses_luca[0], frame_name_0), (poses_user[1], poses_luca[1], frame_name_1)]
    return poses_tuples, poses_luca_norm, switch


def check_correctness(poses_tuple, ex_index, frames):
    """
    Check if the poses are correct by comparing the first frame of the user with the first frame of luca
    :param poses_tuple: list of tuples with the poses of the user and luca
    :param ex_index: index of the exercise
    :param frames: list of frames
    :return: True if the poses are correct, False otherwise
    """
    print("CHECK_CORRECTNESS")

    majinbool = []
    differences = []

    # Compare the angles of the user's poses with the example poses
    for i in range(len(poses_tuple)):
        angle_user = geometry.compute_angle(poses_tuple[i][0], ex_index)
        angle_luca = geometry.compute_angle(poses_tuple[i][1], ex_index)

        print(angle_luca, angle_user)
        result, angle_diff = geometry.compare_angles(angle_user, angle_luca)
        majinbool.append(result)
        differences.append(angle_diff)

    if not majinbool:
        print("Error in the comparison of the angles")
        return

    if False in majinbool:
        for i, val in enumerate(majinbool):
            if not val:
                print("The wrong pose is in frame: ", i)
                print("The angle difference is: ", differences[i])
    else:
        print("Correctly executed exercise, all poses are correct")

    frame_0 = Image.open(poses_tuple[0][2])
    frame_1 = Image.open(poses_tuple[1][2])

    frames[0] = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
    frames[1] = cv2.cvtColor(frames[1], cv2.COLOR_BGR2RGB)

    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].imshow(frame_0)
    axarr[0, 1].imshow(frames[0])
    axarr[1, 0].imshow(frame_1)
    axarr[1, 1].imshow(frames[1])
    plt.show()


def show_frames(video_path, frame_list):
    print("SHOW_FRAMES")
    video = cv2.VideoCapture(video_path)
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frames // 30)  # Step for frame sampling (process at most num_frames frames)
    counter = 0

    for i in range(0, frames, step):
        if counter >= 30:
            break

        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame_video = video.read()
        if not ret:
            break

        if counter in frame_list:
            j = frame_list.index(counter)
            tmp_frame = cv2.resize(frame_video, (480, 720))
            cv2.imshow("Frame " + str(j), tmp_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        counter += 1

    video.release()


def save_frames(video_path, frame_list):
    print("SAVE_FRAMES")
    video = cv2.VideoCapture(video_path)
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_names = []
    step = max(1, frames // 30)  # Step for frame sampling (process at most num_frames frames)
    counter = 0
    c = 0

    for i in range(0, frames, step):
        if counter >= 30:
            break

        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame_video = video.read()
        if not ret:
            break

        if counter in frame_list:
            cv2.imwrite("frame_" + str(c) + ".jpg", frame_video)
            frames_names.append("frame_" + str(c) + ".jpg")
            c += 1
        counter += 1

    video.release()
    return frames_names
