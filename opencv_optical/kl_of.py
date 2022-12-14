import numpy as np
import cv2 as cv
from PIL import Image
import glob
import os


def load_image_gray(imfile):
    img = cv.imread(imfile,cv.IMREAD_GRAYSCALE)
    return img

def load_image_rgb(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[:,:,:3]
    img =cv.cvtColor(img,cv.COLOR_RGB2BGR)
    #print(img.shape)
    gray =cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    return gray,img

def load_image(imfile,mode):
    if mode:
        img = load_image_rgb(imfile)
    else:
        img = load_image_gray(imfile)
    return img

def save_res(bgr, save,i):
    cv.imwrite(os.path.join(save,str(i).zfill(5)+'.png'), bgr)

def prepare_pts(edges):
    points = []
    width, height = edges.shape
    for i in range(width):
        for j in range(height):
            if edges[i][j]>254: #edge = 255
                points.append([[i,j]])
    #print(points)
    return np.array(points,dtype=np.float32)

def run(data, point_path, save,mode):
    images = glob.glob(os.path.join(point_path, '*.png'))
    images = sorted(images)

    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

    lk_params = dict( winSize  = (5, 5),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS , 10, 0.01))
    color = np.random.randint(0, 255, (100, 3))
    #print(images)
    image1 = load_image(images[0],mode)
    p0 = cv.goodFeaturesToTrack(image1,mask = None,**feature_params)
    id = 0

    #mask = np.zeros_like(rgb1)
    for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1,mode)
            image2 = load_image(imfile2,mode)
            #print(os.path.join(point_path,imfile1.split("_")[-1]))
            edges = load_image(os.path.join(point_path,imfile1.split("/")[-1]),False)
            hsv = np.zeros((376,672,3))
            hsv[...,1]=255
            p0=prepare_pts(edges)
            #print(p0.shape)
            p1, st, err = cv.calcOpticalFlowPyrLK(image1, image2, p0, None, **lk_params)
            #p0 = cv.goodFeaturesToTrack(image1, mask = None, **feature_params)
            #print(p0.shape)
            if p1 is not None:
                good_new = p1#[st==1]
                good_old = p0#[st==1]
            w,h = edges.shape
            flow = np.zeros((w,h,2))
            for i in range(len(good_new)):
                x = int(good_old[i][0][0])
                y = int(good_old[i][0][1])
                flow[x][y][0] = good_old[i][0][0] - good_new[i][0][0]
                flow[x][y][1] = good_old[i][0][1] - good_new[i][0][1]
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            hsv = np.float32(hsv)
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            width, height,_ = bgr.shape
            for i in range(width):
                for j in range(height):
                    if bgr[i][j][0]<1 and bgr[i][j][1]<1 and bgr[i][j][2]<1: #edge = 255
                        bgr[i][j] = [255,255,255]


            # p1, st, err = cv.calcOpticalFlowPyrLK(image1, image2, p0, None, **lk_params)
            # if p1 is not None:
            #     good_new = p1[st==1]
            #     good_old = p0[st==1]
            # for i, (new, old) in enumerate(zip(good_new, good_old)):
            #     a, b = new.ravel()
            #     c, d = old.ravel()
            #     mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            #     rgb2 = cv.circle(rgb2, (int(a), int(b)), 5, color[i].tolist(), -1)
            # img = cv.add(rgb2, mask)

            # cv.imshow('frame', img)
            # k = cv.waitKey(30) & 0xff
            # if k == 27:
            #     break
            save_res(bgr,save,id)
            # p0 = good_new.reshape(-1, 1, 2)
            #break
            id = id+1


if __name__ == "__main__":
    point_path = "result/edge_res"
    data_path= "../people/images"
    save_path = "result/kl_res"
    run(data_path, point_path, save_path,False)#mode True = rgb, False = gray




