import av
import numpy as np
from OpenGL.GL import *
from fractions import Fraction

ffps={
    '59.94' : av.utils.Fraction(60000,1001),
    '119.88': av.utils.Fraction(120000,1001),
    '23.98' : av.utils.Fraction(24000,1001),#120000,5005
    '24.00' : av.utils.Fraction(90000,3750),
    '25.00' : av.utils.Fraction(90000,3600),
    '29.97' : av.utils.Fraction(30000,1001),
    '30.00' : av.utils.Fraction(90000,3000),
    '47.95' : av.utils.Fraction(48000,1001),
    '50.00' : av.utils.Fraction(90000,1800),
    '59.94' : av.utils.Fraction(60000,1001),
    '60.00' : av.utils.Fraction(60000,1000),
    '119.88': av.utils.Fraction(120000,1001),
    '120.00': av.utils.Fraction(6000,500)
    }


def fps2frac(fps):
    if isinstance(fps, str) and fps in ffps.keys():
        return ffps[fps]
    elif isinstance(fps, str) and ":" in fps:
        num=int(fps.split(':')[0])
        den=int(fps.split(':')[1])
        return av.utils.Fraction(num,den)
    elif (isinstance(fps, float) or isinstance(fps, int)) and f"{fps:.2f}" in ffps.keys():
        return ffps[f"{fps:.2f}"]
    elif isinstance(fps, av.utils.Fraction):
        return fps


class FrameRecorderAV:
    def __init__(self,output_path,width,height,
                 codec='h264',
                 fps='29.97',
                 options={'c:v':'libx264', 
                          'preset':'slow',
                          'crf':'22'}
                ):
        """
        Video encoder based on PyAV
        :param output_path: Encoded video file path
        :param width: output video width
        :param height: output video height
        :param codec: encoding codec identifier. Options: 'h264' | 'hvec
        :param fps: output video fps
        :param options; options passed to encoder. may include any option understandable by ffmepg
            'pixel_format':  pixel format usually 'yuvj420p'
            'bitrate': average encoding bitrate. Can be specified as a number & unit, e.g. '10M' or '1K'
            )
        :param device: ignored
        """
        assert codec in av.codecs_available
        self.width=width
        self.height=height
        self.output_path=output_path
        self.codec=codec ## av.Codec(codec,)
        self.backend='av'
        _pxformat=options.pop("pixel_format","yuvj420p")
        twitchfactor=[0.06,0.08,0.1,0.15,0.18]  ## quality from lowest to highest
        width*height*float(fps)*twitchfactor[-1]
        _bitrate=options.pop("bitrate",None)
        ## if_bitrate is a string, convert it
        if isinstance(_bitrate, str):
            try:
                _bitrate=int(_bitrate)
            except:
                multiplier=_bitrate[-1]
                if multiplier in ['Mm']:
                    _bitrate=int(_bitrate[:-1])*1000000
                if multiplier in ['Kk']:
                    _bitrate=int(_bitrate[:-1])*1000
        #if isinstance(fps, str):
        #    self._fps=ffps[fps]
        self._fps=fps2frac(fps)

        self.container = av.open(output_path, mode="w")
        self.stream = self.container.add_stream(codec, rate=self._fps,options=options)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = _pxformat
        if not _bitrate is None:
            self.stream.bit_rate = _bitrate

    def _setup_gl(self):
        pass

    def putframe(self,data,*args,**kwargs):
        ## see https://github.com/PyAV-Org/PyAV/issues/596 for hardware encoding
        ## in 1080p, getting the rgb buffer from opengl and converting to yuv420 (without encoding)
        ## induces a frame drop from ~160 fps to ~65 fps
        ## encoding induces a frame drop to ~18/19 fps
        ## not event worth doing rgb to yuv in hardware
        if isinstance(data,int):
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, data)
            buffer=glGetTexImage(GL_TEXTURE_2D,0,GL_RGB,GL_UNSIGNED_BYTE)
            ##buffer=glReadPixels(0,0,self.width, self.height,GL_RGB,GL_UNSIGNED_BYTE)
            #from PIL import Image
            #Image.frombytes('RGB', (self.width,self.height),buffer).show()
            frame=av.VideoFrame.from_ndarray(np.frombuffer(buffer,dtype=np.ubyte).reshape(self.height,self.width,3))
        elif isinstance(data, tuple) and len(data)==2:
            assert(self.stream.pix_fmt=='nv12')
            frame=av.VideoFrame.from_dlpack(data,format='nv12')
            #from PIL import Image
            #y=data[0][::2,::2]
            #u=data[1][:,::2]
            #v=data[1][:,1::2]
            #Image.merge('YCbCr', (Image.fromarray(np.uint8(y)), 
            #                      Image.fromarray(np.uint8(u)),
            #                      Image.fromarray(np.uint8(v)))).show()
        
        
        elif isinstance(data,np.ndarray):
            frame=av.VideoFrame.from_ndarray(data)
        elif isinstance(data,bytes):
            frame=av.VideoFrame.from_ndarray(np.frombuffer(data,dtype=np.ubyte).reshape(self.height,self.width,3))
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        # Flush stream
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()