# -*- coding: utf-8 -*-

"""
@date: 2023/9/20 下午4:39
@file: main.py
@author: zj
@description: https://mnn-docs.readthedocs.io/en/latest/inference/python.html

Usage:
    $ python py/classify.py mobilenet_demo/ILSVRC2012_val_00049999.JPEG mobilenet_demo/mobilenet_v1.mnn

"""

import os
import cv2
import sys
import MNN
import time

import numpy as np


# import MNN

def data_preprocess(image):
    # cv2 read as bgr format
    image = image[..., ::-1]
    # change to rgb format
    image = cv2.resize(image, (224, 224))
    # resize to mobile_net tensor size
    image = image - (103.94, 116.78, 123.68)
    image = image * (0.017, 0.017, 0.017)
    # preprocess it
    # HWC -> CHW
    image = image.transpose((2, 0, 1))
    # change numpy data type as np.float32 to match tensor's format
    image = image.astype(np.float32)

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
    interpreter.resizeTensor(input_tensor, (1, 3, 224, 224))
    interpreter.resizeSession(session)

    return input_tensor, session, interpreter


if __name__ == '__main__':
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    data = data_preprocess(img)

    model_path = sys.argv[2]
    input_tensor, session, interpreter = model_init(model_path)

    start = time.perf_counter()
    # cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
    tmp_input = MNN.Tensor((1, 3, 224, 224), MNN.Halide_Type_Float, data, MNN.Tensor_DimensionType_Caffe)
    print(tmp_input.getShape(), tmp_input.getNumpyData().shape)
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session)
    # constuct a tmp tensor and copy/convert in case output_tensor is nc4hw4
    tmp_output = MNN.Tensor((1, 1001), MNN.Halide_Type_Float, np.ones([1, 1001]).astype(np.float32),
                            MNN.Tensor_DimensionType_Caffe)
    output_tensor.copyToHostTensor(tmp_output)
    end = time.perf_counter()
    time = (end - start) * 1000.
    print("forward time: %fms" % time)

    print("expect 983")
    print(tmp_output.getShape())
    print("output belong to class: {}".format(np.argmax(tmp_output.getData())))
