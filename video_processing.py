import numpy as np
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip


def apply_sharpening(input_path):
    print("APPLY_SHARPENING")
    output_path = input_path.split('.')[-2] + "_sharpened.mp4"
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Build the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Define the gaussian blur kernel
    gaussian_blur_kernel = np.array([[1, 2, 1],
                                     [2, 4, 2],
                                     [1, 2, 1]]) / 16

    # Define the sharpening kernel
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the gaussian blur kernel
        blurred = cv2.filter2D(frame, -1, gaussian_blur_kernel)

        # Apply the sharpening kernel
        sharpened = cv2.filter2D(blurred, -1, sharpening_kernel)

        out.write(sharpened)

    cap.release()
    out.release()

    return output_path


def save_5secs(video_path):
    print("SAVE_5SECS")
    # Upload the video
    video = VideoFileClip(video_path)

    # Compute the start and end time for the 5 seconds video
    duration = video.duration
    if duration < 5:
        print("Video is too short")
        exit()
    start_time = (duration / 2) - 2.5
    end_time = (duration / 2) + 2.5

    # Cut the video
    short_video = video.subclip(start_time, end_time)

    # Save the new video
    short_video.write_videofile(video_path.split('.')[-2] + "_5s.mp4", codec="libx264")

    return video_path.split('.')[-2] + "_5s.mp4"


def retrieve_video(poses_user, data_poses_luca):
    print("FIND_VIDEO_RETRIEVE")
    min_diff = np.inf
    exercise_name = ""

    for exercise, pose in data_poses_luca.items():
        diff = np.linalg.norm(np.array(pose) - np.array(poses_user))
        if diff < min_diff:
            min_diff = diff
            exercise_name = exercise

    if exercise_name == "":
        print("Exercise not found")
        return None

    return exercise_name


def play_video(video_path):
    print("PLAY_VIDEO")
    video = cv2.VideoCapture(video_path)
    delay = 1000 // int(video.get(cv2.CAP_PROP_FPS))
    if (video.isOpened() == False):
        print("Error opening video file")
    while True:
        ret, frame = video.read()
        if not ret:
            break
        cv2.imshow("Video", frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
