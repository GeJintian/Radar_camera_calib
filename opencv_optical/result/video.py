import os
import cv2
import glob


images = glob.glob(os.path.join('dense_res', '*.png'))
images = sorted(images)
video = cv2.VideoWriter('dense_res/edge_result.avi',cv2.VideoWriter_fourcc(*'MJPG'),10,(672,376))
for i in images:
    img = cv2.imread(i)
    video.write(img)
video.release()