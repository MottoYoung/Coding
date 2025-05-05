import cv2


def read_video(video_path):
    clip=cv2.VideoCapture(video_path)
    frames=[]
    while True:
        ret, frame = clip.read()
        if not ret:
            break
        frames.append(frame)
    return frames
def write_video(output_video_frames,output_video_path):
    # print("type of output video frames",output_video_frames[0])
    # print("length of output video frames",len(output_video_frames))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()