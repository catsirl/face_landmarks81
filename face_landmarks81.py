# -*-coding:utf-8-*
import sys
import os
import dlib
import glob
# from skimage import io
import numpy as np
import cv2
import Tkinter as tk
import tkFileDialog


# 返回一个特征点的横纵坐标
def _shape_to_np(shape):
    xy = []
    for i in range(81):
        # shape.part(i)是第i个特征点, x和y属性分别对应该特征点的横纵坐标
        xy.append((shape.part(i).x, shape.part(i).y,))
    xy = np.asarray(xy, dtype='float32')
    return xy

# 新建.pts数据文件保存81个特征点标记
def write_landmarks_to_file(pts_path, lmarks):
    f = open(pts_path, 'w', 0)
    f.write('# landmarks need to be in the form:\n')
    f.write('# x    y\n')
    for lmark in lmarks:
        for i in range(81):
            lmark_x = '%.3f' % lmark[i, 0]
            lmark_y = '%.3f' % lmark[i, 1]
            f.write(lmark_x + ' ' + lmark_y + '\n')
    f.close()



# 输入图像路径
root = tk.Tk()
root.withdraw()
image_path = tkFileDialog.askopenfilename()


# 以彩色模式读入图像
img = cv2.imread(image_path, 1)

# data文件路径
this_path = os.path.dirname(os.path.abspath(__file__))
this_path = this_path.split('\\')
# 文件路径改为'/'分隔
predictor_path = '/'.join(this_path)
predictor_path = predictor_path + '/shape_predictor_81_face_landmarks.dat'



# dlib训练好的人脸检测器
detector = dlib.get_frontal_face_detector()
# dlib训练好的人脸特征检测器
predictor = dlib.shape_predictor(predictor_path)

# .pts文件路径
# 文件路径以'\'分隔
this_path = os.path.dirname(os.path.abspath(__file__))
this_path = this_path.split('\\')
# 文件路径改为'/'分隔
pts_path = '/'.join(this_path)
file_path = ((image_path.split('/')[-1]).split('.'))[0]
# 获得.pts路径
pts_path = pts_path + '/' + file_path + '.pts'


# save_path路径
this_path = os.path.dirname(os.path.abspath(__file__))
this_path = this_path.split('\\')
# 文件路径改为'/'分隔
save_path = '/'.join(this_path)
# 保存为无损压缩格式.png
save_path = save_path + '/' + file_path + '_81_landmarks.png'


lmarks = []
shapes = []
# 侦测人脸，dets保存着所有人脸检测矩形的左上和右下坐标
dets = detector(img, 0)
# det保存着单个人脸检测矩形的左上和右下坐标
for k, det in enumerate(dets):
    # 特征点保存在shape里面
    shape = predictor(img, det)
    shapes.append(shape)
    xy = _shape_to_np(shape)
    lmarks.append(xy)
    # 展示81个特征点
    landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
    for num in range(shape.num_parts):
        # 画圆圈
        cv2.circle(img, (shape.parts()[num].x, shape.parts()[num].y), 3, (0,255,0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # 画数字
        cv2.putText(img, str(num + 1), (shape.parts()[num].x, shape.parts()[num].y), font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

# 保存81个特征点
write_landmarks_to_file(pts_path, lmarks)
# 展示结果
cv2.imshow('frame', img)
cv2.imwrite(save_path, img)
cv2.waitKey(0)
cv2.destoryAllWindows()


