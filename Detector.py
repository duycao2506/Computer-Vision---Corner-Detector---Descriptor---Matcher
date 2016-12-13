#Cao Khac Le Duy @copyright
# 1351008
# All rights are reserved

import cv2
"""
OpenCV
"""
import numpy as np
"""
Matrix operation library
"""
import time

class Detector(object):
    "Dectector Parent Class"
    def __init__(self):
        self.windowName = ""
        self.img = None
        self.cloneImg = None
        self.matcher = None
        self.callback = self.displayDetected
        pass

    def addUi(self):
        pass

    def detect(self, img):
        if self.img != None:
            self.callback = None
            self.img = img
            self.displayDetected()
            return
        self.img = img
        "Detect"

    def getDetected(self,img):
        pass

    def displayDetected(self):
        kpnts = self.getDetected(self.img)
        if kpnts != None:
            self.cloneImg = cv2.drawKeypoints(self.img, kpnts, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow(self.windowName, self.cloneImg)
        

    def setMatcher(self, matcher):
        self.matcher = matcher
        if self.matcher != None:
            self.matcher.setDetector(self)

    def setWindowName(self, wdName):
        self.windowName = wdName


###################################
#####------------   Blob    ------#
###################################

class BlobDetector(Detector):
    "Blob Method Detector"

    def __init__(self):
        "constructor"
        super(BlobDetector, self).__init__()
        self.params = cv2.SimpleBlobDetector_Params()

        self.params.minDistBetweenBlobs = 0.0
        self.params.minThreshold = 10
        self.params.maxThreshold = 10
        self.params.filterByArea = True
        self.params.minArea = 0
        self.params.filterByCircularity = True
        self.params.minCircularity = 0.0
        self.params.filterByConvexity = True
        self.params.minConvexity = 0.0
        self.params.filterByInertia = True
        self.params.minInertiaRatio = 0.0
        self.callbacks = [self.setMinDistBlob, 
                    self.setAreaMin, 
                    self.setCirMin, 
                    self.setConvexmin, 
                    self.setInertia, 
                    self.setMinThresold, 
                    self.setMaxThresold]
        
    def addUi(self):
        cv2.createTrackbar("Min Dist between blobs",self.windowName, 0, 14,self.callbacks[0])
        cv2.createTrackbar("Min Area",self.windowName, 0, 15, self.callbacks[1])
        cv2.createTrackbar("Min Circularity",self.windowName, 0, 150, self.callbacks[2])
        cv2.createTrackbar("Min Convexity",self.windowName, 0, 150, self.callbacks[3])
        cv2.createTrackbar("Min Inertia Ratio",self.windowName, 0, 150, self.callbacks[4])
        cv2.createTrackbar("Min Thresold",self.windowName, 0, 15, self.callbacks[5])
        cv2.createTrackbar("Max Thresold",self.windowName, 0, 15, self.callbacks[6])

    def detect(self, img):
        Detector.detect(self, img)
        self.displayDetected()
        self.addUi()
        

    def getDetected(self,img):
        detector = cv2.SimpleBlobDetector_create(self.params)
        start = time.time()
        kpnts = detector.detect(img)
        end = time.time()
        print end - start, " Time consuming "
        return kpnts

    def setMinDistBlob(self,val):
        self.params.minDistBetweenBlobs = val
        if self.callback != None:
            self.callback()
        

    # def setFilterByColor(self,val):
        # pass

    # def setBlobColor(self,val):
    #     pass

    def setAreaMin(self,val):
        self.params.minArea = val*100
        if self.callback != None:
            self.callback()

    def setCirMin(self,val):
        self.params.minCircularity = val/100
        if self.callback != None:
            self.callback()
        pass

    def setConvexmin(self, val):
        self.params.minConvexity = val/10
        if self.callback != None:
            self.callback()

    def setInertia(self, val):
        self.params.minInertiaRatio = val/1000
        if self.callback != None:
            self.callback()
    
    def setMinThresold(self,val):
        self.params.minThreshold = val * 10
        if self.callback != None:
            self.callback()
        
    
    def setMaxThresold(self, val):
        self.params.maxThreshold = val * 10
        if self.callback != None:
            self.callback()

    def displayDetected(self):
        Detector.displayDetected(self)
        pass
    


###################################
#####------------   Harris  ------#
###################################

class HarrisDetector(Detector):
    "Harris Method Detector"

    def __init__(self):
        super(HarrisDetector, self).__init__()
        self.minBlocksize = 2
        self.minkdsize = 1
        self.kdsize = self.minkdsize
        self.blocksize = self.minBlocksize
        self.kfreeval = 0
        self.callbacks = [self.setKDSize, self.setKFreeVal, self.setBlockSize]

    def addUi(self):
        cv2.createTrackbar("Derivative K Size",self.windowName, 0, 14, self.callbacks[0])
        cv2.createTrackbar("Harris K Value",self.windowName, 0, 30, self.callbacks[1])
        cv2.createTrackbar("Block size",self.windowName, 0, 15, self.callbacks[2])

    def detect(self, img):
        Detector.detect(self, img)
        self.displayDetected()
        self.addUi()
        pass

    def getDetected(self, img):
        dest = None
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kpnts = None
        self.cloneImg = np.ones((img.shape[0],img.shape[1],img.shape[2]), dtype = "uint8")*img
        if self.blocksize > 0:
            start = time.time()
            dest = cv2.cornerHarris(gray, self.blocksize, self.kdsize*2 + 1, self.kfreeval, cv2.BORDER_REFLECT101)
            dest = cv2.dilate(dest,None)    
            kpntss = zip(*np.where(dest > 0.01 * dest.max()))
            kpnts = [cv2.KeyPoint(y,x,10) for x,y in kpntss]      
            end = time.time()
            print end - start, " Time consuming "                           
        else:
            self.cloneImg = img
        return kpnts
        
    def displayDetected(self):
        Detector.displayDetected(self)
        pass

    def setBlockSize(self, val):
        self.blocksize = val + self.minBlocksize
        if self.callback != None :
            self.callback()
    
    def setKDSize(self, val):
        self.kdsize = val + self.minBlocksize
        if self.callback != None :
            self.callback()

    def setKFreeVal(self, val):
        self.kfreeval = val / 100.0
        if self.callback != None :
            self.callback()




###################################
#####------------   DoG     ------#
###################################

class DoGDetector(Detector):
    "DoG Method detecting"
    def __init__(self):
        super(DoGDetector, self).__init__()
        self.thresold = 0
        self.callbacks = [self.setThresold]

    def addUi(self):
        cv2.createTrackbar("Thresold",self.windowName, 0, 20, self.callbacks[0])

    def detect(self, img):
        Detector.detect(self,img)
        self.displayDetected()
        self.addUi()

    def setThresold(self,vale):
        self.thresold = vale/50.0
        if self.callback != None :
            self.callback()

    def getDetected(self, img):
        kpntss = []
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        start = time.time()
        listlayers = self.getListOfLayers(gray,8)
        differencesLayers = self.getDifferenceofLayers(listlayers)
        conditionalMat = np.zeros(gray.shape,dtype=bool)
        for i in range(1,len(differencesLayers)-1):
            for m in range(1,gray.shape[0]-1):
                for n in range(1,gray.shape[1]-1):
                    if conditionalMat[m][n]:
                        conditionalMat[m][n] = false
                    else:
                        conditionalMat[m][n] = (self.isLargerThanLocals(m,n,gray)) and (differencesLayers[i][m][n] > differencesLayers[i+1][m][n] + self.thresold) and (differencesLayers[i][m][n] > differencesLayers[i-1][m][n] + self.thresold)
            kpntss += zip(*np.where(conditionalMat == True))
        kpnts = [cv2.KeyPoint(y,x,10) for x,y in kpntss]
        end = time.time()
        print end - start, " Time consuming "
        return kpnts
    
    def isLargerThanLocals(self, x,y,grayimg):
        comp = grayimg[x-1:x+2,y-1:y+2]
        a = np.ones((3,3), dtype = np.float32)*grayimg[x,y]
        # print a, " AND ", comp, "AND", np.all(a-comp >= 0)
        return np.all((a-comp) >= 0)
        


    def getDifferenceofLayers(self,listlayers):
        result = [listlayers[i+1]-listlayers[i] for i in range(len(listlayers)-1)]
        return result

    def getListOfLayers(self, grayImg, numLayers):
        listlayer = [self.gaussianDerivative(i,grayImg) for i in range(3,numLayers+3,2)]
        return listlayer

    def gaussianDerivative(self, kernelSize, grayImg):
        kernel = cv2.getGaussianKernel(kernelSize,1)
        grayImg = np.asanyarray(grayImg,dtype= np.float32)
        dest = cv2.filter2D(grayImg, -1, kernel)
        return dest

    def displayDetected(self):
        Detector.displayDetected(self)
        pass
    
