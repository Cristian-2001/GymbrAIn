import numpy as np
import cv2


def find_min_vel(poses, keypoint, num_min=2, th=0.5):
    """
    Find the frames with the minimum velocity based on pose data.

    Args:
        poses (list): A list of pose data, where each pose is a tensor of shape (1, 17, 2).
        keypoint (int): The index of the keypoint to use for velocity calculation.
        num_min (int): The number of frames with the minimum velocity to find. Default is 2.
        th (float): The threshold for the velocity calculation. Default is 0.5.

    Returns:
        min_frames (list): A list of indices of the frames with the minimum velocity.
    """
    print("FIND_MIN_VEL")
    l_dist = []  # List to store distances between consecutive frames
    min_frames = []  # List to store indices of frames with minimum velocity

    # Calculate distances between consecutive frames
    for i in range(len(poses)):
        if poses[i][0][keypoint] == (0, 0):
            continue
        if i == 0:
            dist = np.linalg.norm(poses[i][0][keypoint] - poses[i + 1][0][keypoint])
        else:
            dist = np.linalg.norm(poses[i][0][keypoint] - poses[i - 1][0][keypoint])

        l_dist.append(dist)

    # print(lDist)
    counter = 0
    # Find the frames with the minimum velocity
    while len(min_frames) < num_min and counter < len(l_dist):
        minimum = l_dist.index(min(l_dist))
        # If the list is empty, append the first minimum
        if len(min_frames) == 0:
            min_frames.append(minimum)
            l_dist[minimum] = float('inf')
            continue
        # Check if the frame is too similar to the previous one
        if np.linalg.norm((poses[minimum][0][keypoint] - poses[min_frames[-1]][0][keypoint])) > th \
                and np.linalg.norm((poses[minimum][0][11] - poses[min_frames[-1]][0][11])) < 0.5:
            print(np.linalg.norm((poses[minimum][0][keypoint] - poses[min_frames[-1]][0][keypoint])))
            min_frames.append(minimum)
        l_dist[minimum] = float('inf')
        counter += 1

    if counter == len(l_dist):
        print("Not enough frames with different velocities")
    return min_frames


def camera_calibration(index):
    print("CAMERA_CALIBRATION")
    if index == 0:
        calibration = np.load('camera_coeffs/coefficients_bench.npz')
    elif index == 1:
        calibration = np.load('camera_coeffs/coefficients_lat.npz')
    elif index == 2 or index == 3:
        calibration = np.load('camera_coeffs/coefficients_biceps_triceps_pushup.npz')
    else:
        print("ERROR: Invalid index")
        return None

    camera_matrix = calibration['mtx']
    dist_coeffs = calibration['dist']

    return camera_matrix, dist_coeffs


def pixel_to_3d(point, camera_matrix, dist_coeffs):
    print("PIXEL_TO_3D")
    # Use only x and y coordinates
    point_2d = np.array([point[0], point[1]], dtype=np.float32).reshape(1, 1, 2)
    # Use cv2.undistortPoints to obtain normalized coordinates
    normalized_point = cv2.undistortPoints(point_2d, camera_matrix, dist_coeffs)
    # Add z = 1 coordinate to get a 3D point
    return np.array([normalized_point[0, 0, 0], normalized_point[0, 0, 1], 1.0])


def compute_angle(pose, exercise_index):
    print("COMPUTE_ANGLE")
    camera_matrix, dist_coeffs = camera_calibration(exercise_index)

    point1_3d = pixel_to_3d(pose[6], camera_matrix, dist_coeffs)
    vertex_3d = pixel_to_3d(pose[8], camera_matrix, dist_coeffs)
    point3_3d = pixel_to_3d(pose[10], camera_matrix, dist_coeffs)

    # Compute the vectors between the points
    vector1 = point1_3d - vertex_3d
    vector2 = point3_3d - vertex_3d

    # Compute the angle between the vectors
    angle_rad = np.arccos(np.dot(vector2, vector1) / (np.linalg.norm(vector2) * np.linalg.norm(vector1)))
    angle_deg = 180 - np.degrees(angle_rad)

    return angle_deg


def compare_angles(angle_user, angle_luca):
    print("COMPARE_ANGLES")
    angle_diff = angle_user - angle_luca
    if abs(angle_diff) < 20:
        return True, angle_diff
    else:
        return False, angle_diff
