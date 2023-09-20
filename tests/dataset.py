# -*- coding: utf-8 -*-

"""
@date: 2023/9/19 下午5:22
@file: dataset.py
@author: zj
@description: 
"""

from utils.datasets import TensorDataset

val_txt = ''
input_width = ''
input_height = ''

val_dataset = TensorDataset(val_txt, input_width, input_height, False)
