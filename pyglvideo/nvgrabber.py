import math
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import fractions

import PyNvVideoCodec as nvc
#from .nvutils import get_frame_data
from .common import FrameInfo


class FrameGrabberNV(object):
    def __init__(self,**kwargs):
        self.width  = None             ## width of frame
        self.height = None             ## height of frame
        self.frames_per_sec=None       ## frac
        self.frame_info=FrameInfo(pts=-1,time_base=fractions.Fraction(0,1),key=0)#avtime=0,key=0)
        self.first_pts=-1
        self.backend='av'
        self.hwaccel=kwargs.pop('hwaccel',None)
        self.is_hw_owned=kwargs.pop('is_hw_owned',False) if self.hwaccel=='cuda' else False

    def __frameiter(self):
        for packet in self._nv_dmx:
            for frame in self._nv_dec.Decode(packet):
                #frame_data = get_frame_data(decoded_frame, use_device_memory)
                self.frame_info=FrameInfo(pts=frame.timestamp,
                                 time_base=frame.time_base,
                                 #avtime=frame.time,
                                 key=frame.key_frame)
                yield frame

    def load(self, filename,**kwargs):
        self.name=filename
        self._nv_dmx = nvc.CreateDemuxer(filename=filename)
        self._caps = nvc.GetDecoderCaps(
            gpuid=0,
            codec=self._nv_dmx.GetNvCodecId(),
            chromaformat=self._nv_dmx.ChromaFormat(),
            bitdepth=self._nv_dmx.BitDepth()
            )
        if "num_decoder_engines" in self._caps:
            print("Number of NVDECs:", self._caps["num_decoder_engines"])
        self._nv_dec = nvc.CreateDecoder(gpuid=0,
                               codec=self._nv_dmx.GetNvCodecId(),
                               usedevicememory=1)
        self.frames=self.__frameiter()
        f=next(self.frames)
        self.width  = self._nv_dmx.Width()
        self.height = self._nv_dmx.Height()
        self.frames_per_sec=self._nv_dmx.FrameRate ##self._video.average_rate
        self.num_frames=self._video.frames
        self.frame_info=FrameInfo(pts=-1,
                                  time_base=self._video.time_base,
                                  #avtime=0,
                                  key=0)
        self.seek(0.0)
        self.first_pts=self.frame_info.pts
        ## frames_per_sec is a fraction at that stage, 
        ## but we want it to be an irreducible fraction (FrameRate),
        ## with numerator=stream time base
        ## and  denominator=frame duration in stream time base units
        fps_den=1/(self.frames_per_sec*self.frame_info.time_base)
        fps_num=1/self.frame_info.time_base
        #self.frames_per_sec=FrameRate(fps_num,fps_den)
        self.frames_per_sec=fps_num/fps_den

    @property
    def movieduration(self):
        '''returns the duration in seconds. fractions.Fraction'''
        return self._container.duration/av.time_base
    
    @property
    def movieposition(self):
        '''returns current position in seconds. fractions.Fraction'''
        return (self.frame_info.pts - self.first_pts)/(self.frame_info.time_base.denominator)

    @property
    def movieidx(self):
        ## returns the index of current frame.assumes constant frame rate
        ## consistent with avidemux / ffmpeg output
        ##return (self.frame_info.pts - self.first_pts)/(self.frames_per_sec.denominator)
        return round(self.movieposition*self.frames_per_sec)

    @property
    def frameduration(self):
        return float(1/self.frames_per_sec)
    
    @property
    def planes(self):
        return self.decoded.planes

    def position_from_index(self,idx):
        #ts= idx*self.frames_per_sec.denominator+self.first_pts
        #return float(ts*self._video.time_base)
        # more accurate version?
        return idx/self.frames_per_sec#-self.first_pts*self.frame_info.time_base
    
    def seek(self,seek_time_sec=0.0,exact_frame=True):
        ## "seeking to absolute frame number is not supported by any known format" (pyAV documentation)
        ## more generally, seeking to specific frame requires indexing or approximation with average rate
        ## after seeking, the next frame will be the closest to requested pts.
        ## However, in our code, we read two extra times
        seek_time_sec=max(0, min(seek_time_sec,self.movieduration))
        pts=int(seek_time_sec/self._video.time_base)+self.first_pts-1
        ## dont't seek if we are close to requested frame
        if (pts<self.frame_info.pts) or ((pts-self.frame_info.pts)//self.frames_per_sec.denominator)>25:
            #print(f"Small seek ahead - rather decode than seek")
            self._container.seek(pts,backward=True,any_frame = False,stream=self._video) ## as we specify the stream, we must provide pts in stream timebase
            self.readframe()
            if not exact_frame:
                return
        ahead=0
        while self.frame_info.pts<=pts and ahead<1000: ## may be too long! depends on GOP size
            self.readframe()
            ahead+=1
    
    def readframe(self):
        try:
            self.decoded=next(self._frames)
            self.px_format=self.decoded.format.name
            return True
        except StopIteration:
            self._frames=self.__frameiter()
            return False
