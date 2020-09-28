import numpy as np
import cv2
import os
import sys
from imutils import paths
from timeit import default_timer as timer
start = timer()

def dhash(image, hashSize=9):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize, hashSize))
    blurred=cv2.medianBlur(resized,hashSize)
    diff = blurred[:, 1:] > blurred[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

dir_path = list(paths.list_images('C:\\Future work\\Codes\\Python intermediate\\aug_dogs-20200504T131640Z-001 (2)'))   #Change this to the input dataset path
out_dir='C:\\Future work\\Codes\\Python intermediate\\updated' #Change this path to the folder you want output in

if sys.platform != "win32":
	dir_path = [x.replace("\\", "") for x in dir_path]

count=0
arr2=[]

def unique(hash_mini,arr):
    for i in range(8):
        if hash_mini[i] in arr2:
            return False
    return True

for p in dir_path:
    image = cv2.imread(p)
    if image is None:
    	continue
    hash_mini=[]
    img_rotate_90_clockwise = image.copy()
    for i in range(4):
        img_rotate_90_clockwise = cv2.rotate(img_rotate_90_clockwise, cv2.ROTATE_90_CLOCKWISE)
        imageHash = dhash(img_rotate_90_clockwise)
        hash_mini.append(imageHash)
        img_flip=cv2.flip(img_rotate_90_clockwise,1)
        imageHash = dhash(img_flip)
        hash_mini.append(imageHash)
    
    if unique(hash_mini,arr2):
        least=min(hash_mini)
        arr2.append(least)
        count+=1
        cv2.imwrite(out_dir+'\\'+str(count)+'.jpg',image)

elapsed_time = timer() - start

print("Number of images in the updated dataset is: "+str(count))
print("Time Taken: "+str(elapsed_time)+"s")