# -*- coding: utf-8 -*-

"""
@date: 2023/9/20 下午4:39
@file: main.py
@author: zj
@description: https://mnn-docs.readthedocs.io/en/latest/inference/python.html
"""

import os
import cv2
import sys
import MNN
import time

import numpy as np


# import MNN

def data_preprocess(image):
    image = cv2.resize(image, (352, 352))
    # preprocess it
    # HWC -> CHW
    image = image.transpose((2, 0, 1))
    # change numpy data type as np.float32 to match tensor's format
    image = image.astype(np.float32)
    image = image.reshape((1, 3, 352, 352)) / 255

    return image


def model_init(model_path):
    assert os.path.isfile(model_path), model_path

    interpreter = MNN.Interpreter(model_path)
    interpreter.setCacheFile('.tempcache')

    # 配置执行后端，线程数，精度等信息；key-vlaue请查看API介绍
    config = {}
    config['precision'] = 'low'  # 当硬件支持（armv8.2）时使用fp16推理
    config['backend'] = 0  # CPU
    config['numThread'] = 4  # 线程数

    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    interpreter.resizeTensor(input_tensor, (1, 3, 352, 352))
    interpreter.resizeSession(session)

    return input_tensor, session, interpreter


# sigmoid函数
def sigmoid(x):
    return 1. / (1 + np.exp(-x))


# tanh函数
def tanh(x):
    return 2. / (1 + np.exp(-2 * x)) - 1


# nms算法
def nms(dets, thresh=0.45):
    # dets:N*M,N是bbox的个数，M的前4位是对应的（x1,y1,x2,y2），第5位是对应的分数
    # #thresh:0.3,0.5....
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 求每个bbox的面积
    order = scores.argsort()[::-1]  # 对分数进行倒排序
    keep = []  # 用来保存最后留下来的bboxx下标

    while order.size > 0:
        i = order[0]  # 无条件保留每次迭代中置信度最高的bbox
        keep.append(i)

        # 计算置信度最高的bbox和其他剩下bbox之间的交叉区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算置信度高的bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 求交叉区域的面积占两者（置信度高的bbox和其他bbox）面积和的必烈
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留ovr小于thresh的bbox，进入下一次迭代。
        inds = np.where(ovr <= thresh)[0]

        # 因为ovr中的索引不包括order[0]所以要向后移动一位
        order = order[inds + 1]

    output = []
    for i in keep:
        output.append(dets[i].tolist())

    return output


if __name__ == '__main__':
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    pred = []

    # 输入图像的原始宽高
    H, W, _ = img.shape

    data = data_preprocess(img)

    model_path = sys.argv[2]
    input_tensor, session, interpreter = model_init(model_path)

    start = time.perf_counter()
    # cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
    tmp_input = MNN.Tensor((1, 3, 352, 352), MNN.Halide_Type_Float, data, MNN.Tensor_DimensionType_Caffe)
    print(tmp_input.getShape(), tmp_input.getNumpyData().shape)
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session)
    print(output_tensor.getShape())

    # constuct a tmp tensor and copy/convert in case output_tensor is nc4hw4
    tmp_output = MNN.Tensor((1, 6, 22, 22), MNN.Halide_Type_Float, np.ones([1, 6, 22, 22]).astype(np.float32),
                            MNN.Tensor_DimensionType_Caffe)
    output_tensor.copyToHostTensor(tmp_output)
    end = time.perf_counter()
    time = (end - start) * 1000.
    print("forward time: %fms" % time)

    print(tmp_output.getShape())
    feature_map = tmp_output.getNumpyData().squeeze()

    # 输出特征图转置: CHW, HWC
    feature_map = feature_map.transpose(1, 2, 0)
    # 输出特征图的宽高
    feature_map_height = feature_map.shape[0]
    feature_map_width = feature_map.shape[1]

    thresh = 0.45
    # 特征图后处理
    for h in range(feature_map_height):
        for w in range(feature_map_width):
            data = feature_map[h][w]

            # 解析检测框置信度
            obj_score, cls_score = data[0], data[5:].max()
            score = (obj_score ** 0.6) * (cls_score ** 0.4)

            # 阈值筛选
            if score > thresh:
                # 检测框类别
                cls_index = np.argmax(data[5:])
                # 检测框中心点偏移
                x_offset, y_offset = tanh(data[1]), tanh(data[2])
                # 检测框归一化后的宽高
                box_width, box_height = sigmoid(data[3]), sigmoid(data[4])
                # 检测框归一化后中心点
                box_cx = (w + x_offset) / feature_map_width
                box_cy = (h + y_offset) / feature_map_height

                # cx,cy,w,h => x1, y1, x2, y2
                x1, y1 = box_cx - 0.5 * box_width, box_cy - 0.5 * box_height
                x2, y2 = box_cx + 0.5 * box_width, box_cy + 0.5 * box_height
                x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)

                pred.append([x1, y1, x2, y2, score, cls_index])

    bboxes = nms(np.array(pred))

    # 加载label names
    names = ['box']
    # with open("coco.names", 'r') as f:
    #     for line in f.readlines():
    #         names.append(line.strip())

    print("=================box info===================")
    for b in bboxes:
        print(b)
        obj_score, cls_index = b[4], int(b[5])
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])

        # 绘制检测框
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
        cv2.putText(img, names[cls_index], (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

    cv2.imwrite("result.jpg", img)
