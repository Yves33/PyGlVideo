import logging
import python_vali as vali      # using VALI
import numpy as np
import math
import fractions

from .common import FrameInfo

vali_pixel_formats={
 'y': vali.PixelFormat.Y,
 'rgb': vali.PixelFormat.RGB,
 'nv12': vali.PixelFormat.NV12,
 'yuv420': vali.PixelFormat.YUV420,
 'rgb_planar': vali.PixelFormat.RGB_PLANAR,
 'bgr': vali.PixelFormat.BGR,
 'yuv444': vali.PixelFormat.YUV444,
 'yuv444_10bit': vali.PixelFormat.YUV444_10bit,
 'yuv420_10bit': vali.PixelFormat.YUV420_10bit,
 'undefined': vali.PixelFormat.UNDEFINED,
 'rgb_32f': vali.PixelFormat.RGB_32F,
 'rgb_32f_planar': vali.PixelFormat.RGB_32F_PLANAR,
 'yuv422': vali.PixelFormat.YUV422,
 'p10': vali.PixelFormat.P10,
 'p12': vali.PixelFormat.P12
}

supported_pixel_formats=['nv12','rgb','yuv420','yuv444']
class OnePassConverter:
    def __init__(self,fmt,w,h):
        self.fmt, self.w, self.h=fmt, w, h
        self.cvt=vali.PySurfaceConverter(gpu_id=0)
        
    def Run(self,src,dst,cc_ctx):
        success,details=self.cvt.Run(src,dst,cc_ctx)
        return success, details

class FrameGrabberVALI(object):
    def __init__(self,**kwargs):
        self.width  = None             ## width of frame
        self.height = None             ## height of frame
        self.frames_per_sec=None       ## frac
        self.frame_info=FrameInfo(pts=-1,time_base=fractions.Fraction(0,1),key=0)#avtime=0,key=0)
        self.first_pts=-1
        self.is_hw_owned=kwargs.pop('is_hw_owned',True) ## by default, we keep data on gpu
        self.tgt_format=kwargs.pop('tgt_format','nv12')
        if not self.is_hw_owned and self.tgt_format not in supported_pixel_formats:
            print(f"Convertion to {self.tgt_format} with is_hw_owned=1 is not supported")
            exit()
        self.backend='vali'

    def load(self, filename,**kwargs):
        '''ffmpeg / pyav uses 1µsec as time unit'''
        self.name=filename
        self.nvDec = vali.PyDecoder(filename, opts={},gpu_id=0)
        den_=round(1/(self.nvDec.Timebase*self.nvDec.Framerate))
        num_=round(self.nvDec.Framerate*den_)
        #self.frames_per_sec=FrameRate(num_,den_)
        self.frames_per_sec=fractions.Fraction(num_,den_)
        assert( math.fabs(1/self.nvDec.Timebase-int(1/self.nvDec.Timebase))<0.05)
        self._pckt_timebase=int(1/self.nvDec.Timebase)
        self.num_frames=self.nvDec.NumFrames
        self.width, self.height = self.nvDec.Width, self.nvDec.Height
        
        tgt_color_space=self.nvDec.ColorSpace if self.nvDec.ColorSpace!=vali.ColorSpace.UNSPEC else vali.ColorSpace.BT_709
        tgt_color_range=self.nvDec.ColorRange if self.nvDec.ColorRange!=vali.ColorRange.UDEF else vali.ColorRange.MPEG
        self.cc_ctx = vali.ColorspaceConversionContext(tgt_color_space,tgt_color_range)
        self.nvCvt=OnePassConverter(self.nvDec.Format,self.nvDec.Width,self.nvDec.Height)
        ## create downloader and buffer for CUDA ->CPU ->GL pipeline
        self.nvDwn = vali.PySurfaceDownloader(0)

        self.first_pts=self.frame_info.pts
        self.eos=False
        assert( math.fabs(1/self.nvDec.Timebase-int(1/self.nvDec.Timebase))<0.05)
        if self.num_frames:
            self._duration=float(self.num_frames/self.frames_per_sec)
        elif self.nvDec.Duration>0:
            self._duration=self.nvDec.Duration
        else:
            try:
                self._duration=self.nvDec.Duration
                hh,mm,ss=self.nvDec.Metadata['video_stream']['DURATION'].split(':')
                self._duration=int(hh)*3600+int(mm)*60+float(ss.replace(',','.'))
            except:
                self._duration=3600 ##(!)
        self.seek(0.0)
        self.first_pts=self.frame_info.pts

    @property
    def movieduration(self):
        '''returns current position in seconds. fractions.Fraction'''
        return self._duration

    @property
    def movieposition(self):
        '''returns current position in seconds. fractions.Fraction'''
        return (self.frame_info.pts - self.first_pts)/(self.frame_info.time_base.denominator)
    
    @property
    def movieidx(self):
        '''returns current frame index'''
        return round(self.movieposition*self.frames_per_sec)
            
    @property
    def frameduration(self):
        '''returns frame duration'''
        return float(1/self.frame_per_sec)
    
    @property 
    def planes(self):
        if self.is_hw_owned: 
            return self.decoded.Planes
        else:
            return self.cpu_buffers
    
    def position_from_index(self,idx):
        #ts= idx/(self.frame_info.time_base*self.frames_per_sec)+self.first_pts
        #return ts*self.frame_info.time_base
        # more accurate version?
        return idx/self.frames_per_sec#-self.first_pts*self.frame_info.time_base

    def seek(self,seek_time_sec=0.0,exact_frame=True, max_frame_delta=30):
        pdata=vali.PacketData()
        seek_time_sec=max(0, min(seek_time_sec, self.movieduration))
        pts=max(0,int(seek_time_sec*self._pckt_timebase)+self.first_pts-1)
        if (pts<self.frame_info.pts) or (pts-self.frame_info.pts)*self.frame_info.time_base >max_frame_delta/self.frames_per_sec:
            logging.debug("large seek ahead")
            seek_ctx=vali.SeekContext(
                                seek_ts=seek_time_sec+self.first_pts/self._pckt_timebase, 
                                #mode=vali.SeekMode.EXACT_FRAME if exact_frame else vali.SeekMode.PREV_KEY_FRAME
                                )
            self._decode(pdata,seek_ctx)
            self.px_format=self.decoded.Format.name.lower()
            self.frame_info=FrameInfo(pts=pdata.pts,
                                time_base=fractions.Fraction(1,self._pckt_timebase),
                                key=pdata.key)
            return True
        else:
            ahead=0
            while self.frame_info.pts<=pts and ahead<1000 and not self.eos: ## may be too long! depends on GOP size
                self._decode(pdata)
                self.px_format=self.decoded.Format.name.lower()
                self.frame_info=FrameInfo(pts=pdata.pts,
                                    time_base=fractions.Fraction(1,self._pckt_timebase),
                                    key=pdata.key)
                ahead=ahead+1
            return True
        '''
        seek_ctx=vali.SeekContext(
                                seek_ts=seek_time_sec+self.first_pts/self._pckt_timebase, 
                                #mode=vali.SeekMode.EXACT_FRAME if exact_frame else vali.SeekMode.PREV_KEY_FRAME
                                )
        self._decode(pdata,seek_ctx)
        self.px_format=self.decoded.Format.name.lower()
        self.frame_info=FrameInfo(pts=pdata.pts,
                                time_base=fractions.Fraction(1,self._pckt_timebase),
                                #avtime=pdata.pts*self.frames_per_sec.numerator,
                                key=pdata.key)'''

    def _decode(self,*args,**kwargs):
        ## todo test for end of stream
        self.decoded=vali.Surface.Make(self.nvDec.Format,self.nvDec.Width,self.nvDec.Height,0)
        success, info = self.nvDec.DecodeSingleSurface(self.decoded, *args,**kwargs)
        if (not success) or self.decoded.IsEmpty:
            print("failed decoding surface!")
            return success,info
        if self.decoded.Format.name.lower()!=self.tgt_format:
            self.decoded_ = vali.Surface.Make(vali_pixel_formats[self.tgt_format], self.decoded.Width,self.decoded.Height, 0)
            success,info=self.nvCvt.Run(self.decoded, self.decoded_,self.cc_ctx)
            self.decoded=self.decoded_
            if not success:
                print("failed converting surface!")
                return success,info
        if not self.is_hw_owned:
            if self.decoded.Format.name.lower() in ['nv12','cuda']:
                self.yuv_buffer=np.zeros((self.width,3*self.height//2),np.uint8)
                self.nvDwn.Run(self.decoded, self.yuv_buffer)
                if not success:
                    print("failed downloading nv12 surface!")
                    return success,info
                self.cpu_buffers=[self.yuv_buffer.reshape(3*self.height//2,self.width)[:self.height],
                                  self.yuv_buffer.reshape(3*self.height//2,self.width)[self.height:]
                                ]
            elif self.decoded.Format.name.lower() in ['yuv420','yuv420p']:
                self.yuv_buffer=np.zeros(self.width*3*self.height//2,np.uint8)
                success,info=self.nvDwn.Run(self.decoded, self.yuv_buffer)
                if not success:
                    print("failed downloading yuv surface!")
                    return success,info
                sz=self.height*self.width
                self.cpu_buffers=[self.yuv_buffer[:sz].reshape(self.height,self.width),
                                  self.yuv_buffer[sz:sz+sz//4].reshape(self.height//2,self.width//2),
                                  self.yuv_buffer[sz+sz//4:].reshape(self.height//2,self.width//2)
                ]
            elif self.decoded.Format.name.lower()in ['yuv_444','yuv444p']:
                self.yuv_buffer=np.zeros(self.width*3*self.height,np.uint8)
                success,info=self.nvDwn.Run(self.decoded, self.yuv_buffer)
                if not success:
                    print("failed downloading yuv surface!")
                    return success,info
                sz=self.height*self.width
                self.cpu_buffers=[self.yuv_buffer[:sz].reshape(self.height,self.width),
                                  self.yuv_buffer[sz:2*sz].reshape(self.height,self.width),
                                  self.yuv_buffer[2*sz:3*sz].reshape(self.height,self.width)
                                 ]
            elif self.decoded.Format.name.lower()=='rgb':
                self.cpu_buffers=[np.zeros(self.width*3*self.height,np.uint8)]
                success,info=self.nvDwn.Run(self.decoded, self.cpu_buffers[0])
                if not success:
                    print("failed downloading yuv surface!")
                    return success,info
                self.cpu_buffers[0]=self.cpu_buffers[0].reshape(self.height,self.width,3)
                ##Image.fromarray(self.cpu_buffers[0].reshape(2880,5760,3)).show()               
        return success, info

    def readframe(self):
        pdata=vali.PacketData()
        success,info=self._decode(pkt_data=pdata)
        self.px_format=self.decoded.Format.name.lower()
        self.frame_info=FrameInfo(pts=pdata.pts,
                                 time_base=fractions.Fraction(1,self._pckt_timebase),
                                 #avtime=pdata.pts/self.frames_per_sec.numerator,
                                 key=pdata.key)
        if success:
            return True