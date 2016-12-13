#Cao Khac Le Duy @copyright
# 1351008
# All rights are reserved


import sys
import cv2
import numpy as np
from enum import Enum
import time

class Matcher(object):
    def __init__(self):
        self.detector = None
        self.imgs = []
        pass
    
    def match(self,imgs):
        self.imgs = imgs
        pass
    
    def setDetector(self,detector):
        self.detector = detector
        detector.callback = self.doMatch
        pass

    def doMatch(self):
        pass

class SiftMatcher(Matcher):
    def __init__(self):
        super(SiftMatcher,self).__init__()
    

    def match(self,imgs):
        Matcher.match(self,imgs)
        self.doMatch()
        self.detector.addUi()
        pass
    
    def doMatch(self):
        kpts1 = self.detector.getDetected(self.imgs[0])
        kpts2 = self.detector.getDetected(self.imgs[1])
        grays = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in self.imgs]
        
        sift = cv2.xfeatures2d.SIFT_create()
        start = time.time()
        desc = []
        desc = sift.compute(grays,[kpts1, kpts2],desc)
        
        
        bf = cv2.BFMatcher()
        matches = None
        # print (desc[1][0][0])
        matches = bf.knnMatch(desc[1][0],desc[1][1], 2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        end = time.time()
        print end - start, " Time consuming "
        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = None
        img3 = cv2.drawMatchesKnn(self.imgs[0], kpts1, self.imgs[1], kpts2, matches,img3,matchColor = [0,255,0], flags=2)
        cv2.imshow(self.detector.windowName, img3)
    
    

class LbpMatcher(Matcher):
    def __init__(self):
        super(LbpMatcher, self).__init__()
    
    def match(self,imgs):
        Matcher.match(self,imgs)
        self.doMatch()
        self.detector.addUi()
        pass
    
    def doMatch(self):
        kpnts = [self.detector.getDetected(image) for image in self.imgs] 
        grays = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in self.imgs]
        
        start = time.time()
        lbp = LbpDescriptorExtractor()
        desc = [lbp.extractDescriptors(kpnts[0],grays[0]),
                lbp.extractDescriptors(kpnts[1],grays[1])] 
        bf = cv2.BFMatcher()
        
        matches = None
        matches = bf.knnMatch(np.asarray(desc[0],dtype=np.float32),np.asarray(desc[1],dtype=np.float32),2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        end = time.time()
        print end - start, " Time consuming "
        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = None
        img3 = cv2.drawMatchesKnn(self.imgs[0],kpnts[0],self.imgs[1],kpnts[1],matches,img3,matchColor = [0,255,0], flags=2)
        cv2.imshow(self.detector.windowName, img3)

    

class LbpDescriptorExtractor(object):
    "Extract descriptors from keypoints"

    def __init__(self):
        pass

    def extractDescriptors(self, kpnts, img):
        descrpt = [self.buildHistoVector(self.getCellLbp(key,img)) for key in kpnts]
        return descrpt

    def getCellLbp(self, keypoint, img):
        x,y = keypoint.pt
        gapx = len(img) - int(x)
        gapy = len(img[0]) - int(y)
        gapx =  gapx if gapx < 8 else 8
        gapy =  gapy if gapy < 8 else 8
        cell = img[int(x) - gapx : int(x) + gapx, int(y) - gapy : int(y) + gapy]
        return cell

    def buildHistoVector(self, lbpCell):
        vectorLbp = [0]*256
        for i in range(1,len(lbpCell)-1):
            for j in range(1,len(lbpCell[0])-1):
                index = self.getBinaryNumberAt(i,j,lbpCell)
                vectorLbp[index] += 1.0
        return vectorLbp

    def getBinaryNumberAt(self, x, y, ofCell):
        count = 0
        count += ((ofCell[x][y] < ofCell[x-1][y-1]) << 7)
        count += ((ofCell[x][y] < ofCell[x-1][y]) << 6)
        count += ((ofCell[x][y] < ofCell[x-1][y+1]) << 5)
        count += ((ofCell[x][y] < ofCell[x][y+1]) << 4)
        count += ((ofCell[x][y] < ofCell[x+1][y+1]) << 3)
        count += ((ofCell[x][y] < ofCell[x+1][y]) << 2)
        count += ((ofCell[x][y] < ofCell[x+1][y-1]) << 1)
        count += ((ofCell[x][y] < ofCell[x][y-1]) << 0)
        return count