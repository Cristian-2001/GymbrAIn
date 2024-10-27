import numpy as np

import video_processing
import utils
import geometry
import deep_learning

if __name__ == "__main__":
    # List of exercise classes
    class_list = ['bench press', 'lat pulldown', 'barbell biceps curl', 'tricep Pushdown']
    luca_class_list = ['panca', 'lat', 'biceps', 'triceps']

    # Keypoints to use for velocity calculation for each exercise
    keypoint_to_use_vel = [8, 10, 10, 10]

    # Thresholds for velocity calculation for each exercise
    thresholds = [0.2, 0.2, 0.5, 0.5]

    # List to store poses of the user
    stationary_user_poses = []
    stationary_user_poses_norm = []

    # Path to the video file of the user
    video_path = f"videoTest/pancaaldo.mp4"
    # video_path = f"videoTest/lat_1.mp4"

    # Preprocess the video
    video_path = video_processing.save_5secs(video_path)
    video_path = video_processing.apply_sharpening(video_path)

    # Estimate poses from the user's video
    user_poses, user_poses_norm = deep_learning.estimate_poses(video_path)

    # Classify the exercise based on the estimated poses
    ex_index = deep_learning.classify_exercise(user_poses)

    # Print the index of the classified exercise
    print("Exercise's index: " + str(ex_index))

    # Get the name of the classified exercise
    ex_index = int(ex_index[0])
    exercise = class_list[ex_index]

    # Load pose data from a JSON file
    data_luca = utils.load_json("poses5.json")
    data_luca, data_luca_norm = data_luca[0], data_luca[1]

    # Retrieve the video paths for the example exercise
    luca_video_path, luca_video_path_720, exercise_name = video_processing.retrieve_video(user_poses_norm,
                                                                                          data_luca_norm)

    # Check if classification and retrieval give the same exercise
    if ex_index != luca_class_list.index(exercise_name):
        print("The classified exercise is different from the example exercise")
        exit()

    # Print the name of the executed exercise and the example video path
    print("Performed exercise: " + exercise)
    print("Example video: " + luca_video_path)

    # Play the example video
    video_processing.play_video(luca_video_path_720)

    # Find frames with minimum velocity based on the estimated poses
    min_vel_frames = geometry.find_min_vel(user_poses_norm, keypoint=keypoint_to_use_vel[ex_index],
                                           th=thresholds[ex_index])
    user_frames = utils.save_frames(video_path, min_vel_frames)

    # Uncomment the following line to show the frames with minimum velocity
    # utils.show_frames(video_path, min_vel_frames)

    # Find the corresponding frames in the video
    frame_list = utils.find_frames(video_path, min_vel_frames)

    # Append the poses with minimum velocity to the list
    for index in min_vel_frames:
        stationary_user_poses.append(user_poses[index][0])
        stationary_user_poses_norm.append(user_poses_norm[index])

    # Get the poses of the example exercise
    poses_luca = data_luca[exercise_name]
    poses_luca[0] = np.array(poses_luca[0])
    poses_luca[1] = np.array(poses_luca[1])
    poses_luca_norm = data_luca_norm[exercise_name]
    poses_luca_norm[0] = np.array(poses_luca_norm[0])
    poses_luca_norm[1] = np.array(poses_luca_norm[1])

    # Find the most similar poses between the user and the example
    poses_tuples, poses_luca_norm, switch = utils.find_same_poses(stationary_user_poses, stationary_user_poses_norm,
                                                                  poses_luca, poses_luca_norm, exercise_name)

    # Compare the angles of the user's poses with the example poses
    utils.check_correctness(poses_tuples, ex_index, frame_list)
