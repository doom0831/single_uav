# import cv2
# import glob

# path = "./outputs/UE4 and Airsim/DDPG_test/results/DE/*.png" 
# result_name = 'output.mp4'

# frame_list = sorted(glob.glob(path))
# print("frame count: ",len(frame_list))

# fps = 30
# shape = cv2.imread(frame_list[0]).shape # delete dimension 3
# size = (shape[1], shape[0])
# print("frame size: ",size)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(result_name, fourcc, fps, size)

# for idx, path in enumerate(frame_list):
#     frame = cv2.imread(path)
#     # print("\rMaking videos: {}/{}".format(idx+1, len(frame_list)), end = "")
#     current_frame = idx+1
#     total_frame_count = len(frame_list)
#     percentage = int(current_frame*30 / (total_frame_count+1))
#     print("\rProcess: [{}{}] {:06d} / {:06d}".format("#"*percentage, "."*(30-1-percentage), current_frame, total_frame_count), end ='')
#     out.write(frame)

# out.release()
# print("Finish making video !!!")

import cv2
import os
import re

def sort_images(image_name):
    # Extract the number part of the filename using regular expression
    numbers = re.findall(r'\d+', image_name)
    if numbers:
        return int(numbers[0])
    return 0

def images_to_video(image_folder, output_video, fps=30, image_format='.png'):
    # Load all images and sort them by the number in their filenames
    images = [img for img in os.listdir(image_folder) if img.endswith(image_format)]
    images.sort(key=sort_images)  # Sort using the custom sort function

    # Read the first image to get the size
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

# Usage
images_to_video('./outputs/UE4 and Airsim/DDPG_test/results/DE', 'output_video.mp4', fps=3)


