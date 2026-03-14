# 工具库  
from re import X
import setup_vsim
import ivs
import numpy as np
import cv2
import os
import pprint
import time
import math
# 自建函数
from ivs import Vector3r,Pose

if __name__ == "__main__":
      # 连接初始化
    client = ivs.VSimClient()  # 创建连接
    client.confirmConnection()  # 检查连接
    client.reset()
    client.simFlushPersistentMarkers()  # 清空画图
    #参考路径点可视化
    x=[0,1,2,4,10]
    y=[0,1,2,4,10]
    points=[]
    for i in range(len(x)): 
        temp=Vector3r(x[i], y[i],-1)
        points.append(temp)
    print("---------simPlotLineStrip Start------------")
    print(points)
    client.simPlotPoints(points, color_rgba=[1.0, 0.0, 0.0, 1.0],size=10,duration=-1.0, is_persistent=True)
