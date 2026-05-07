import numpy as np
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from PIL import Image
import fractions

from .common import FrameInfo
            
class FrameGrabberPIL(object):
    def __init__(self,**kwargs):
        self.width  = None
        self.height = None
        self.status = 0
        self.frames_per_sec=None
        self.fdelay=None
        self.num_frames=None
        self.duration=None
        self.planes=None
        self.frame_info=FrameInfo(pts=-1,time_base=fractions.Fraction(0,1),key=0)#avtime=0,key=0)
        self.first_pts=-1
        self.backend='pil'

    def load(self, filename,**kwargs):
        self.name=filename
        self._image=Image.open(filename)
        self.width,self.height  = self._image.size
        self.planes=[np.array(self._image.convert('RGB'))]
        self.px_format='rgb'
        self.frames_per_sec=fractions.Fraction(30000,1001) ## could be parsed from kwargs
        self.frame_info=FrameInfo(pts=0,
                                  time_base=fractions.Fraction(self.frames_per_sec.numerator,1),
                                  #avtime=0,
                                  key=0)
        self.fdelay=1/self.frames_per_sec
        self.num_frames=65535 ## could be parsed from kwargs
        self.duration=float(self.num_frames*self.frames_per_sec)
        self.first_pts=0

    @property 
    def movieduration(self):
        if self.num_frames<0:
            return -1
        return float(self.num_frames/self.frames_per_sec)
    
    @property 
    def movieposition(self):
        if self.num_frames<0:
            return -1
        return (self.frame_info.pts//self.frames_per_sec.denominator)/float(self.frames_per_sec)
    
    @property
    def movieidx(self):
        return (self.frame_info.pts - self.first_pts)/self.frames_per_sec.denominator

    @property
    def frameduration(self):
        return float(1/self.frames_per_sec)
    
    def position_from_index(self,idx):
        return idx/self.frames_per_sec
    
    def seek(self,seek_time_sec,exact_frame=True):
        _fidx=np.ceil(seek_time_sec*self.frames_per_sec)
        if _fidx>self.num_frames:
            raise EOFError("Seeked past last frame")
        self.frame_info.pts=_fidx*self.frames_per_sec.denominator
        #self.frame_info.avtime=self.frame_info.seconds
        pass

    def readframe(self):
        # fake timestamp adjustment
        if self.movieidx>=self.num_frames:
            raise EOFError("Attempt to read past last frame")
        self.frame_info.pts+=float(self.frames_per_sec.denominator)
        #self.frame_info.avtime=self.frame_info.seconds