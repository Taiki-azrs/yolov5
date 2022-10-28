import cv2
import numpy as np
import glob
import os
from multiprocessing import Pool, cpu_count
os.makedirs('images', exist_ok=True)
os.makedirs('labels', exist_ok=True)
#for i,mov_path in enumerate(glob.glob('move/*')):
def run(p_arg):
    i,mov_path=p_arg
    print(mov_path)
    cap=cv2.VideoCapture(mov_path)
    cnt=0
    while(cap.isOpened):
        ret, frame = cap.read()
        if ret==False:
            break
        cv2.imwrite('images/{}-{}.png'.format(i,cnt),frame)
        f = open('labels/{}-{}.txt'.format(i,cnt),'w')
        f.write('0 0.518 0.509 0.304 0.673\n')
        f.close()
        print('{}-{}'.format(i,cnt))
        cnt+=1
def main():
    p_arg = [ [i,mov_path] for i,mov_path in enumerate(glob.glob('move/*'))]
    pool_obj = Pool(cpu_count())
    pool_obj.map(run,p_arg)
if __name__ == "__main__":
    main()
