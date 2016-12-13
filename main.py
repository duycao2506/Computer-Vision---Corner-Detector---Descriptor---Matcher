#Cao Khac Le Duy @copyright
# 1351008
# All rights are reserved

import sys
import cv2
from enum import Enum
from Detector import HarrisDetector, BlobDetector, DoGDetector, Detector
from Matcher import SiftMatcher, LbpMatcher, Matcher
import numpy as np

class Args(Enum):
    "Keys for controlling"
    detect = "d"
    match = "m"
    harris = "harris"
    blob = "blob"
    dog = "dog"
    lbp = "lbp"
    sift = "sift"

class DetectorWrapper(object):
    "Wrapping Detecting Method"
    
    def __init__(self):
        self.detector = None
        self.imgs = []
        self.task = Args.detect

    def setDetector(self, detector, matcher = None):
        "setter for detector"
        if self.detector == None:
            self.detector = detector
            self.detector.setMatcher(matcher)
            self.detector.setWindowName ("HW2")

    def addImgs(self, imgs):
        "Add image to wrapper"
        if len(self.imgs) != 0:
            return
        for str in imgs:
            self.imgs.append(cv2.imread(str))

    def setTask(self, task):
        self.task = task

    def doTask(self):
        "do task"
        if self.task == Args.match:
            self.detector.matcher.match(self.imgs)
        else:
            self.detector.detect(self.imgs[0])


def mapping_function(detectorWrapper, args):
    "Map key controls to functionalities"

    imageIndices = [1, 3]
    doId = Args.detect
    if detectorWrapper.detector != None :
        detectorWrapper.doTask()
    
    #set task_
    detectorWrapper.setTask(args[0])
    
    #create matcher
    matcher = None
    imgs = [args[imageIndices[0]]]
    if args[0] == Args.match:
        imgs = args[imageIndices[1]:]
        if args[2] == Args.sift:
            matcher = SiftMatcher()
        elif args[2] == Args.lbp:
            matcher = LbpMatcher()

    #create detector
    if args[0] == Args.harris or args[1] == Args.harris:
        detectorWrapper.setDetector(HarrisDetector(),matcher)
        print "HARRIS"
    elif args[0] == Args.blob or  args[1] == Args.blob:
        detectorWrapper.setDetector(BlobDetector(),matcher)
    elif args[0] == Args.dog or args[1] == Args.dog:
        detectorWrapper.setDetector(DoGDetector(), matcher)

    #last steps
    detectorWrapper.addImgs(imgs)
    detectorWrapper.doTask()
    
        
        


def display_help_lines():
    "display help string"
    strin = "Run this command from command line: python main.py -i [options]\n"\
            "#####The [options] stands for: \n"\
            "- harris image.jpg detect key points using harris algorithm and show the keypoints in original image.\n"\
            "- blob image.jpg - detect key points using blob algorithm and show the keypoints in original image.\n"\
            "- dog image.jpg detect key points using DoG Algorithm and show keypoints in original image.\n"\
            "- m harris sift image1.jpg image2.jpg match and show results of image1 and image2 using Harris detector and SIFT descriptor.\n"\
            "- m dog sift image1.jpg image2.jpg - match and show results of image1 and image2 using DoG detector and SIFT descriptor.\n"\
            "- m blob sift image1.jpg image2.jpg - match and show results of image1 and image2 using using Blob detector and SIFT descriptor.\n"\
            "- m harris lbp image1.jpg image2.jpg - match and show results of image1 and image2 using Harris detector and LBP descriptor.\n"\
            "- m dog lbp image1.jpg image2.jpg - match and show results of image1 and image2 using DoG detector and LBP descriptor.\n"\
            "- m blob lbp image1.jpg image2.jpg - match and show results of image1 and image2 using Blob detector and LBP descriptor."
    return strin


def main(arv):
    "Main Loop"
    wrapper = DetectorWrapper()
    if (arv[0] == "-i"):
        key = 25
        while key!= 27:
            print "OK"
            if chr(key) == 'h':
                print display_help_lines()
            else:
                mapping_function(wrapper, arv[1:])
            key = cv2.waitKey(0)
    

    cv2.destroyAllWindows()
            
    print arv

if __name__ == "__main__":
    main(sys.argv[1:])
