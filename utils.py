import cv2
import tkinter as tk
import numpy as np



def GetScreenCenter():
    root = tk.Tk()
    return root.winfo_screenwidth()//2,root.winfo_screenheight()//2

def AdaptSize(img):
    # 视频、图片过大直接1/2
    center_x, center_y = GetScreenCenter()
    img_h, img_w, _ = img.shape
    if img_h > center_y * 2 or img_w > center_x * 2:
        img = cv2.resize(img, (img_w // 2, img_h // 2))
    return img

def CentralShow(win_name,img,stop):
    center_x, center_y = GetScreenCenter()
    img=AdaptSize(img)
    img_h,img_w,_=img.shape
    t_x, t_y = (center_x - img_w // 2), (center_y - img_h // 2)
    cv2.imshow(win_name, img)
    cv2.moveWindow(win_name, t_x, t_y)
    if stop:
        cv2.waitKey(0)


def GetClass():
    id2name={}
    with open('coco_id2name') as f:
        for idx,line in enumerate(f):
            idd,name=line.split(':')
            id2name[int(idd)]=name.strip()
    return id2name

def ShowPicResult(cv_mat, win_name, scores, labels, boxes,id2name,colors,stop=False):

    for score, label, box in zip(scores, labels, boxes):
        if score > 0.5:
            x1, y1, x2, y2 = box
            color=colors[label.item()].squeeze(0)
            color=tuple(int(a) for a in color)
            # print(id2name)
            # print(label.item())
            # exit()
            cv_mat = cv2.putText(cv_mat, id2name[label.item()], (int((x2-x1)/2+x1), int((y2-y1)/2)+y1), cv2.FONT_HERSHEY_TRIPLEX, .5,
                                 color, 1)
            # left_top point and right bottom point
            cv_mat = cv2.rectangle(cv_mat, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    CentralShow(win_name, cv_mat,stop)


# eg:
# win_name='centered video'
# video = cv2.VideoCapture('1604400296582.mp4')

# CentralShow(win_name,video)
# ShowVideo(video)











