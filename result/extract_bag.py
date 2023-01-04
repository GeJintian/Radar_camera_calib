import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import cv2
from cv_bridge import CvBridge
import yaml


depth = '/scoutmini_alpha/zed2i/zed_node/depth/depth_registered'
cam = '/scoutmini_alpha/zed2i/zed_node/rgb/image_rect_color'
radar = '/ti_mmwave/radar_scan_pcl_0'
cam_info = '/scoutmini_alpha/zed2i/zed_node/left/camera_info'

bag = rosbag.Bag('cam_radar.bag','r')
bridge = CvBridge()
#info = bag.get_type_and_topic_info()
depth_data = bag.read_messages(depth)
cam_data = bag.read_messages(cam)
radar_data = bag.read_messages(radar)
cam_info = bag.read_messages(cam_info)

# depth_data
count = 0
for topic,msg,t in depth_data:

    #pct = pc2.read_points(msg)
    d_image = bridge.imgmsg_to_cv2(msg,"mono16")
    #d_image = d_image.copy()*1000.0
    #d_image = d_image.astype(np.uint16)
    #print(msg.header.stamp.secs)
    #pct_np=np.array(list(pct))
    #break
    cv2.imwrite("depth/"+str(msg.header.stamp)+'.png',d_image)
    #np.save("radar/frame"+str(count).zfill(4)+'.npy',pct_np)
    count = count +1
bag.close()
