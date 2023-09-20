# -*- coding: utf-8 -*-

"""
@date: 2023/9/19 下午2:55
@file: detector.py
@author: zj
@description: 
"""

import torch

from module.detector import Detector

if __name__ == "__main__":
    # model = Detector(80, False)
    model = Detector(80, True)
    test_data = torch.rand(1, 3, 352, 352)

    outpus = model(test_data)
    print(outpus.shape)

    # # Export
    # torch.onnx.export(model,  # model being run
    #                   test_data,  # model input (or a tuple for multiple inputs)
    #                   "./test.onnx",  # where to save the model (can be a file or file-like object)
    #                   export_params=True,  # store the trained parameter weights inside the model file
    #                   opset_version=11,  # the ONNX version to export the model to
    #                   do_constant_folding=True)  # whether to execute constant folding for optimization
