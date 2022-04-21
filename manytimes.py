import numpy as np
import cv2 as cv
import copy
imgO = cv.imread("ImageDecayed60%.jpg")
h,w,c = imgO.shape
routine = 10
img = copy.deepcopy(imgO)

#IMG DECAYED GENERATED#
for i in range(routine):
    R = copy.deepcopy(img[:,:,0])
    G = copy.deepcopy(img[:,:,1])
    B = copy.deepcopy(img[:,:,2])

    scale = 4
    D1 = np.zeros(0,dtype=np.uint8)
    D2 = np.zeros(0,dtype=np.uint8)
    D3 = np.ones(scale*scale,dtype=np.uint8)
    for x in range(0,scale):
        D1 = np.append(D1,np.arange(scale,dtype=np.uint8).reshape(1,scale))
        D2 = np.append(D2,x*np.ones(scale,dtype=np.uint8))

    for X in range(0,h,scale):
        for Y in range(0,w,scale):
            D1t = np.array([],dtype=np.uint8)
            D2t = np.array([],dtype=np.uint8)
            D3t = np.array([],dtype=np.uint8)
            Rtt = R[X:X+scale,Y:Y+scale].reshape(scale*scale,1)
            Gtt = G[X:X+scale,Y:Y+scale].reshape(scale*scale,1)
            Btt = B[X:X+scale,Y:Y+scale].reshape(scale*scale,1)
            Rt = np.array([[]])
            Gt = np.array([[]])
            Bt = np.array([[]])
            flag = 0
            for x in range(0,scale):
                for y in range(0,scale):
                    if R[X+x,Y+y] != 0 and G[X+x,Y+y] != 0 and B[X+x,Y+y] != 0:
                        D1t = np.append(D1t,D1[scale*x+y])
                        D2t = np.append(D2t,D2[scale*x+y])
                        D3t = np.append(D3t,D3[scale*x+y])
                        Rt = np.append(Rt,[Rtt[scale*x+y]])
                        Gt = np.append(Gt,[Gtt[scale*x+y]])
                        Bt = np.append(Bt,[Btt[scale*x+y]])
                        flag = 1
            if flag == 0: continue
            Dt = np.array([D1t,D2t,D3t])
            aR=np.linalg.pinv(Dt@(Dt.T))@Dt@Rt
            aG=np.linalg.pinv(Dt@(Dt.T))@Dt@Gt
            aB=np.linalg.pinv(Dt@(Dt.T))@Dt@Bt
            for x in range(0,scale):
                for y in range(0,scale):
                    if R[X+x,Y+y] == 0 and G[X+x,Y+y] == 0 and B[X+x,Y+y] == 0:
                        R[X+x,Y+y] = aR[0]*x+aR[1]*y+aR[2]
                        G[X+x,Y+y] = aG[0]*x+aG[1]*y+aG[2]
                        B[X+x,Y+y] = aB[0]*x+aB[1]*y+aB[2]
    img = np.zeros((h,w,c),dtype=np.uint8)

    #Fix Over

    for X in range(0,h):
        for Y in range(0,w):
            img[X][Y][0] = R[X][Y]
            img[X][Y][1] = G[X][Y]
            img[X][Y][2] = B[X][Y]
        print("OK")


cv.imwrite("ImageFixed10Times60%.jpg",img)
