import os
import cv2

def make_video(original_video_path, frame_folder_path, video_name):
    images = [img for img in sorted(os.listdir(frame_folder_path)) if img.endswith(".png") or img.endswith(".jpg")]

    frame = cv2.imread(os.path.join(frame_folder_path, images[0]))
    height, width, layers = frame.shape

    videoCapture = cv2.VideoCapture(original_video_path)
    fps = int(videoCapture.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    video = cv2.VideoWriter(video_name, fourcc, fps * 4, (width,height))

    for image in sorted(images):
        video.write(cv2.imread(os.path.join(frame_folder_path, image)))

    video.release()