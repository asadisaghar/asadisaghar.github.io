---
layout: post
title: Label images using OpenCV object tracker
---

Everyone knows how lazy I am at doing repetitive stuff, to the point that the mere thought of annotating several hundred images [per class] for the object detection project was about to jeopardize the entire project I was so excited about... that, until one day over lunch when I was complaining to my colleague and he said: "but why don't you have the OpenCV object tracker you've been using for another project to label video frames for you?". Why did I not think about it before? So, I started looking for a labeling tool to do this for me.  A couple hours and a few blog posts and tools that were doing only part of the job, I decided that I have spent more time googling than it takes ton actually build the thing. I added the keyboard interactions along the way, as I started using it and needed a little more functionality, but the core took less than an hour to write...and then I had to push myself for a few more days before I started writing about it!

I am using OpenCV 3.3.0 on Python 2.7. It was a challenge to get the _VideoCapture_ function to work properly on my Ubuntu 16.04 machine and apparently it wasn't just me. This could be tricky, especially if you're using anaconda to begin with. To be perfectly honest, I still don't know how it was magically solved, but I disabled my Anaconda python and then followed [this](https://www.learnopencv.com/install-opencv3-on-ubuntu//) instruction to install OpenCV. This happened to be the place I learned how to write a simple object tracker as well. 

Now about the video labeler. What it does is, it accepts the path to your video, where you want to save the frames as jpeg files, where you want to save the labels (with a csv format convertible to TFrecord as mentioned in my previous post), the rate at which you want to dump frames into image files and the label for the object class, as parameters. It should be possible to run it on live videos as well, but I haven't tested that option. Once the video starts running, you need to draw 4 boxes covering the object you want to track, I then use the openCV _multi-tracker_ to follow those points, but the actual bounding box is the area covered by the 4 selected regions. The reason I had to use 4 points rather than simply having the user draw one bbox to begin with is that a single tracker cannot handle zoom-in/zoom-outs of the video and change of scale, the size of the box remains constant and the bbox soon looses accuracy. Even though using 4 leading points is a very simple solution to this issue, it's still not enough. The algorithm can loose track of the object(s) for many reasons, or if you need to use videos that are not captured with a specific object in mind, there may be more than one object class appearing/disappearing along the way. I decide to solve this by giving the master labeler the option to pause the video and help/fix the tracking points, and now that we're at it, how about making it possible to change the label and move on to a whole new object class altogether? In the end, the script saves the labeled video as well.

[![Here is a sample video of a labeling process during which I change the objects I want to track multiple times](https://img.youtube.com/vi/Pa6ARjV8wy0/0.jpg)](https://www.youtube.com/watch?v=Pa6ARjV8wy0)

This is a tool I am using heavily these days, so it's still very much a work in progress. I would like to make a more user friendly interface for it and many other ideas, but for now it is what it is and it makes my life (or annotating training set images) much easier.

```python
from math import *
import numpy as np
import cv2
import sys
import os
import argparse
import time

def read_bboxes(image):
    # choose the corners (or edges) of the tracking bbox
    bbox1 = cv2.selectROI('tracking', image)
    bbox2 = cv2.selectROI('tracking', image)
    bbox3 = cv2.selectROI('tracking', image)
    bbox4 = cv2.selectROI('tracking', image)
    return bbox1, bbox2, bbox3, bbox4

```
