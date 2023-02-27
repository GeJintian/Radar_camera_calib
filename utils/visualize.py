import numpy as np
import cv2

from .helpers import World2Cam, Pos_transform, build_matrix

def seg_mask(rgb,seg):
    width, height = seg.shape
    #print(np.unique(seg))
    for i in range(width):
        for j in range(height):
            if seg[i][j]!=12: #edge = 255
                rgb[i][j] = [0,0,0]

def viz_optical(img, flo, seg_result, i):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    #flo = flow_viz.flow_to_image(flo)
    hsv = np.zeros_like(img)
    hsv[...,1]=255
    mag, ang = cv2.cartToPolar(flo[..., 0], flo[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    img = np.uint8(img)
    gray =cv2.cvtColor(img[:,:,[2,1,0]],cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,200)
    #mask(bgr,edges)
    seg_mask(bgr,seg_result)
    
    bgr = bgr[:,:,[2,1,0]]
    img_flo = np.concatenate([img, bgr], axis=0)
    img_flo = img_flo[:, :, [2,1,0]]
    #img_flo = print_minmax(img_flo,flo,seg_result)

    cv2.imwrite('result/'+str(i).zfill(5)+'.png', img_flo)

def viz_mask(mask,i):
    height, width = mask.shape
    img = np.ones((height,width,3))*255
    for h in range(height):
        for w in range(width):
            if mask[h][w] == 1:
                img[h][w] = [0,0,0]

    cv2.imwrite('result/'+str(i).zfill(5)+'.png',img)

def print_minmax(rgb,flo,seg_mask):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255,255,255)
    thickness = 2
    flow = np.sqrt(np.square(flo[...,0])+np.square(flo[...,1]))
    width,height=seg_mask.shape
    for i in range(width):
        for j in range(height):
            if seg_mask[i][j] !=12:
                flow[i][j] = 0
    max_idx = np.unravel_index(np.argmax(flow, axis=None), flow.shape)
    for i in range(width):
        for j in range(height):
            if seg_mask[i][j] !=12:
                flow[i][j] = 10000
    min_idx = np.unravel_index(np.argmin(flow, axis=None), flow.shape)
    #print(max_idx)
    rgb = rgb.astype(np.uint8).copy()
    #print(rgb.dtype)
    rgb = cv2.putText(rgb,"("+str(int(flo[max_idx[0]][max_idx[1]][0]))+","+str(int(flo[max_idx[0]][max_idx[1]][1]))+")",(max_idx[0],max_idx[1]+width+1),font,fontScale,color,thickness,cv2.LINE_AA)
    
    rgb = cv2.putText(rgb,"("+str(int(flo[min_idx[0]][min_idx[1]][0]))+","+str(int(flo[min_idx[0]][min_idx[1]][1]))+")",(min_idx[0],min_idx[1]+width+1),font,fontScale,color,thickness,cv2.LINE_AA)
    
    return rgb

def viz_pts(mask, imfile, P_r, t, K):
    
    point_size = 1
    point_color = (0,0,255)
    thickness = 4

    M_t = build_matrix(t)
    height, width = mask.shape
    img = np.ones((height,width,3))*255
    for h in range(height):
        for w in range(width):
            if mask[h][w] == 1:
                img[h][w] = [0,0,0]

    for i in P_r:
        p_c = Pos_transform(M_t,i)
        u,v = np.round(World2Cam(K, p_c)).astype(int)
        if v[0] < 0 or v[0] > height-1 or u[0] < 0 or u[0] > width-1:
            print("Out of index")
        # if mask[v[0]][u[0]] != 1:
        #     print("Not 1")
        cv2.circle(img,(u[0],v[0]), point_size, point_color, thickness)
        #print(u[0],v[0])
        
    #print('result/Carla/coarse/'+imfile.split('/')[-1])
    cv2.imwrite('result/Carla/coarse/'+imfile.split('/')[-1],img)