import logging
import av
import fractions

from .common import FrameInfo
from av.codec.hwaccel import HWAccel, hwdevices_available

class FrameGrabberAV(object):
    def __init__(self,**kwargs):
        self.width  = None             ## width of frame
        self.height = None             ## height of frame
        self.frames_per_sec=None       ## frac
        self.frame_info=FrameInfo(pts=-1,time_base=fractions.Fraction(0,1),duration=-1,key=0)
        self.first_pts=-1
        self.backend='av'
        self.hwaccel=kwargs.pop('hwaccel',None)
        self.convert_rgb=kwargs.pop('tgt_format',None)
        self.is_hw_owned=kwargs.pop('is_hw_owned',False) if self.hwaccel=='cuda' else False

    def __frameiter(self):
        try:
            for frame in self._container.decode(video=0):
                self.frame_info=FrameInfo(pts=frame.pts,
                                    time_base=frame.time_base,
                                    duration=frame.duration,
                                    key=frame.key_frame)
                yield frame
        except:
            self.eos=True
            yield None

    def load(self, filename,**kwargs):
        self.name=filename
        if self.hwaccel and self.hwaccel in hwdevices_available():
            hwaccel = HWAccel(device_type=self.hwaccel, 
                              allow_software_fallback=False,
                              is_hw_owned=self.is_hw_owned)
            self._container = av.open(filename,hwaccel=hwaccel,**kwargs)
        else:
            self._container = av.open(filename,**kwargs)
        self._video = self._container.streams.video[0]
        try:
            self._audio = self._container.streams.audio[0]
        except:
            self._audio=None
        self._video.thread_type = 'AUTO'
        self._frames=self.__frameiter()
        self.width  = self._video.width
        self.height = self._video.height
        self.frames_per_sec=self._video.base_rate
        #self.frames_per_sec=self._video.average_rate
        #self.frames_per_sec=self._video.guessed_rate
        #self._frame_duration=1/self.frames_per_sec
        self.frame_info=FrameInfo(pts=-1,
                                  time_base=self._video.time_base,
                                  duration=-1,
                                  key=0)
        self.eos=False
        self.seek(0.0)
        self.first_pts=self.frame_info.pts
        self.px_format=self.decoded.format.name

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
        return 1/self.frames_per_sec
    
    @property
    def planes(self):
        return self.decoded.planes

    def position_from_index(self,idx):
        #ts= idx*self.frames_per_sec.denominator+self.first_pts
        #return float(ts*self._video.time_base)
        # more accurate version?
        return idx/self.frames_per_sec#-self.first_pts*self.frame_info.time_base
    
    def seek(self,seek_time_sec=0.0,exact_frame=True, max_frame_delta=30):
        ## "seeking to absolute frame number is not supported by any known format" (pyAV documentation)
        ## more generally, seeking to specific frame requires indexing or approximation with average rate
        ## after seeking, the current frame (self.decoded) is the first frame that comes after seek time.
        ## seek time sec can be provided as float or fraction
        seek_time_sec=max(0, min(seek_time_sec,self.movieduration))
        pts=int(seek_time_sec/self._video.time_base)+self.first_pts-1
        ## dont't seek if we are close to requested frame
        if (pts<self.frame_info.pts-self.frame_info.duration) or (pts-self.frame_info.pts)*self.frame_info.time_base >max_frame_delta/self.frames_per_sec:
            logging.debug("large seek ahead")
            self._container.seek(pts,backward=True,any_frame = False,stream=self._video) ## as we specify the stream, we must provide pts in stream timebase
            self.readframe()
            if not exact_frame:
                return True
        ahead=0
        while self.frame_info.pts<=pts and ahead<1000 and not self.eos: ## 1000 may be too long! depends on GOP size
            self.readframe()
            ahead+=1
        return True
    
    def readframe(self):
        try:
            ## because we need to always have something valid in self.decoded
            _decoded=next(self._frames)
            if _decoded:
                if self.convert_rgb:
                    self.decoded=_decoded.to_rgb()
                else:
                    self.decoded=_decoded
                return True
        except StopIteration:
            self._frames=self.__frameiter()
            return False
        return False
