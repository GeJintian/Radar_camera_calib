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
n = 0
for topic,msg,t in radar_data:

    pct = pc2.read_points(msg)
    #d_image = bridge.imgmsg_to_cv2(msg,"32FC1")
    #d_image = np.array(d_image)
    #print(msg)
    # d_image = d_image.copy()*1000.0
    # d_image = d_image.astype(np.uint16)
    #print(np.max(d_image))
    #k = np.array(list(msg.K))
    #print(msg)
    # is_moving = False
    pct_np=np.array(list(pct))
    for i in pct_np:
        if abs(i[4])==0:
            is_moving = True
            n = n + 1
    # np.save("calibration.npy",k)
    #break
    #cv2.imwrite("depth/"+str(msg.header.stamp.secs)+"_"+str(msg.header.stamp.nsecs).zfill(9)+'.png',d_image)
    #np.save("depth/"+str(msg.header.stamp.secs)+"_"+str(msg.header.stamp.nsecs).zfill(9)+'.npy',d_image)
    #np.save("radar/"+str(msg.header.stamp.secs)+"_"+str(msg.header.stamp.nsecs).zfill(9)+'.npy',pct_np)
    # print(is_moving)
    #is_moving = False
    
bag.close()
print(n)