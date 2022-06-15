import cv2 
import numpy as np 
import os
from matplotlib import pyplot as plt
from IPython.display import Image
from google.colab.patches import cv2_imshow
from pyparsing.core import disable_diag

src = cv2.imread('ex2imgo.png')
plt.imshow(src)

def gifDilateErode(img):
  fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
  out = cv2.VideoWriter("gifDilate.avi",fourcc,20.0,(112,150))
  kernelSize = 1
  frames = 0

  while kernelSize <= 5:
      kernel = np.ones((kernelSize, kernelSize), np.uint8)  
      img = cv2.dilate(img, kernel, iterations = 1)
      out.write(img)
      frames += 1
      kernelSize += 1 
  kernelSize = 0
  while kernelSize <= 5:
      kernel = np.ones((kernelSize, kernelSize), np.uint8)  
      img = cv2.erode(img, kernel, iterations = 1)
      out.write(img)
      frames += 1
      kernelSize += 1 

  out.release()

gifDilateErode(src)

if os.path.exists("gifDilate.gif"):
    os.remove("gifDilate.gif")

os.system("ffmpeg -i " + os.getcwd() + "/gifDilate.avi " + os.getcwd() + "/gifDilate.gif")
Image(open("gifDilate.gif", "rb").read())