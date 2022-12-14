import numpy as np
import cv2 as cv
from PIL import Image
import os
import glob


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[:,:,:3]
    img =cv.cvtColor(img,cv.COLOR_RGB2BGR)
    #print(img.shape)
    img =cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    return img

def save_res(edge, save,i):
    cv.imwrite(os.path.join(save,str(i).zfill(5)+'.png'), edge)

def run(data,save):
    images = glob.glob(os.path.join(data, '*.png'))
    images = sorted(images)
    i = 0
    
    for imfile1 in images:
            image = load_image(imfile1)
            blank = np.ones_like(image)*255
            edges = cv.Canny(image,100,200)
            #ret, thresh = cv.threshold(edges, 127, 255, 0)
            #contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            #pic1 = cv.drawContours(image,contours,-1,(0,255,0),3)
            #cv.imshow("contours",pic1)
            save_res(edges,save,i)
            #print(edges.max())
            #break
            i=i+1
if __name__ =='__main__':
    data = "../images"
    save = "result/edge_res"
    run(data,save)
