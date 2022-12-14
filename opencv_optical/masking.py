import numpy as np
import cv2 as cv
from PIL import Image
import os
import glob


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[:,:,:3]
    img =cv.cvtColor(img,cv.COLOR_RGB2BGR)
    #print(img.shape)
    gray =cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    return img,gray

def save_res(rgb, save,i):
    cv.imwrite(os.path.join(save,str(i).zfill(5)+'.png'), rgb)

def mask(rgb,edge):
    bool_edge = edge/255
    width, height = bool_edge.shape
    for i in range(width):
        for j in range(height):
            if bool_edge[i][j]>0.5: #edge = 255
                rgb[i][j] = [0,0,0]
            else:
                rgb[i][j] = [0,0,0]


def run(data,save):
    images = glob.glob(os.path.join(data, '*.png'))
    images = sorted(images)
    i = 0
    for imfile1 in images:
            rgb,gray = load_image(imfile1)
            edges = cv.Canny(gray,100,200)
            mask(rgb,edges)
            save_res(rgb,save,i)
            #print(edges.max())
            i=i+1

if __name__ =='__main__':
    data = "../images"
    save = "result/masking_res"
    run(data,save)