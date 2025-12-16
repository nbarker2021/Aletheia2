
import numpy as np, time, math
class FrameGenerator:
    """Generates synthetic frames (moving dot) as placeholder for Scene8 live output."""
    def __init__(self, w:int,h:int):
        self.w=w; self.h=h
    def __iter__(self):
        return self
    def __next__(self):
        t=time.time()
        x=int((math.sin(t)+1)/2*self.w)
        y=int((math.cos(t)+1)/2*self.h)
        frame=np.zeros((self.h,self.w,3),dtype=np.uint8)
        frame[y-5:y+5,x-5:x+5]=[0,255,0]
        time.sleep(1/30)
        return frame
