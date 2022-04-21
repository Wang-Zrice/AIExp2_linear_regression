import numpy as np
import cv2 as cv
import random
import copy
img = cv.imread("Lena.jpg")
h,w,c = img.shape

mask = np.zeros((h,w,c))

par = 80
for x in range(h):
    for y in range(w):
        rad = random.randint(0,100)  
        if rad>=par:
            mask[x][y] = np.array([1,1,1],dtype=np.uint8)
        else:
            mask[x][y] = np.array([0,0,0],dtype=np.uint8)

imgDecayed = np.multiply(img,mask).astype(np.uint8)            

#IMG DECAYED GENERATED#

R = copy.deepcopy(imgDecayed[:,:,0])
G = copy.deepcopy(imgDecayed[:,:,1])
B = copy.deepcopy(imgDecayed[:,:,2])
#To avoid reference problem, use deepcopy here 

scale = 16

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
                if R[X+x,Y+y] != 0 and G[X+x,Y+y] != 0 and B[X+x,Y+y] != 0: #Dark plot skipped
                    D1t = np.append(D1t,D1[scale*x+y])
                    D2t = np.append(D2t,D2[scale*x+y])
                    D3t = np.append(D3t,D3[scale*x+y])
                    Rt = np.append(Rt,[Rtt[scale*x+y]])
                    Gt = np.append(Gt,[Gtt[scale*x+y]])
                    Bt = np.append(Bt,[Btt[scale*x+y]])
                    flag = 1
        if flag == 0: continue  #If there is no valid pixel, skip this block.
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
imgFixed = np.zeros((h,w,c),dtype=np.uint8)

#LINEAR REGRESSION#

for X in range(0,h):
    for Y in range(0,w):
        imgFixed[X][Y][0] = R[X][Y]
        imgFixed[X][Y][1] = G[X][Y]
        imgFixed[X][Y][2] = B[X][Y]
dif1 = imgDecayed - img
dif2 = imgFixed - img
e=np.linalg.norm(dif1)
e1=np.linalg.norm(dif2)
print(e,e1)      


cv.imwrite("ImageDecayed.jpg",imgDecayed)
cv.imwrite("ImageFixed.jpg",imgFixed)
