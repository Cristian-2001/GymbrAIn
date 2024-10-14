import numpy as np
import cv2
import json

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


def find_same_poses(poses_user, poses_luca):
    """
    Find the most similar poses between the user and luca, it calculates only the first frame because the other
    should adjust by itself
    :param poses_user: user's poses
    :param poses_luca: Luca's poses
    :return: Luca's poses in the same order as the user's poses
    """
    print("FIND_SAME_POSES")
    dist_0 = np.linalg.norm(poses_user[0] - poses_luca[0])
    dist_1 = np.linalg.norm(poses_user[0] - poses_luca[1])
    if dist_1 < dist_0:
        poses_luca[0], poses_luca[1] = poses_luca[1], poses_luca[0]
    return poses_luca


def check_correctness(poses_user, poses_luca, ex_index):
    """
    Check if the poses are correct by comparing the first frame of the user with the first frame of luca
    :param poses_user: user's poses
    :param poses_luca: Luca's poses
    :return: True if the poses are correct, False otherwise
    """
    print("CHECK_CORRECTNESS")

    majinbool = []
    differences = []

    # Compare the angles of the user's poses with the example poses
    for i in range(len(poses_user)):
        angle_user = geometry.compute_angle(poses_user[i][0], ex_index)
        angle_luca = geometry.compute_angle(poses_luca[i], ex_index)

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
