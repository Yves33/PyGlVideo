import logging
from ctypes import c_void_p

import python_vali as vali      # using VALI
import numpy as np
from OpenGL.GL import *
import av
try:
    import pycuda
    PYCUDA_AVAILABLE=True
    try:
        from pycuda.gl import RegisteredBuffer
        PYCUDA_GL_AVAILABLE=True
    except:
        PYCUDA_GL_AVAILABLE=False
except:
    PYCUDA_AVAILABLE=False
    PYCUDA_GL_AVAILABLE=False
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

def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True

class ColorConverterVALI:
    ## https://github.com/NVIDIA/VideoProcessingFramework/issues/326
    def __init__(self, width: int, height: int, device: int):
        """
        Colorspace conversion chain. Used for converting color spaces defined by VPF
        :param width: Video frame width
        :param height: Video frame height
        :param device: GPU device identifier
        """
        self.gpu_id = device
        self.width = width
        self.height = height
        self.chain = []
        self.surfaces=[]

    def add(self, in_format: vali.PixelFormat, out_format: vali.PixelFormat) -> None:
        """
        Adds a conversion step to the chain
        :param in_format: Input VPF pixel format
        :param out_format: Output VPF pixel format
        """
        #yuvSurface= nvc.Surface.Make(nvc.PixelFormat.YUV420,rawSurface.Width(),rawSurface.Height(),0)
        #success,info = self.nvYuv.Run(rawSurface, yuvSurface,self.cc_ctx)
        #self.cvtSurface = nvc.Surface.Make(nvc.PixelFormat.RGB, rawSurface.Width(),rawSurface.Height(), 0)
        #success,info = self.nvCvt.Run(yuvSurface, self.cvtSurface,self.cc_ctx)

        self.surfaces.append(vali.Surface.Make(out_format,self.width,self.height,self.gpu_id))
        #self.chain.append(vali.PySurfaceConverter(in_format, out_format, self.gpu_id))
        self.chain.append(vali.PySurfaceConverter(self.gpu_id))

    def run(self, surface: vali.Surface) -> vali.Surface:
        """
        Runs a chain of color space conversion steps
        :param surface: The surface object that should be converted to a different color space
        :return: A new surface object
        """
        cc_context = vali.ColorspaceConversionContext(vali.ColorSpace.BT_601, vali.ColorRange.MPEG)
        #cc_context = vali.ColorspaceConversionContext(vali.ColorSpace.BT_709, vali.ColorRange.MPEG)
        success,info=self.chain[0].Run(surface,self.surfaces[0], cc_context)
        if len(self.chain)>1:
            for e in range(1,len(self.chain)):
                success,info=self.chain[e].Run(self.surfaces[e-1],self.surfaces[e], cc_context)
        return self.surfaces[-1].Clone()


class FrameRecorderVALI:
    ## mostly copied from wiki and https://github.com/NVIDIA/VideoProcessingFramework/issues/326
    def __init__(self, output_path, width, height,*,
                 codec='hevc',
                 fps= '29.97',
                 options={'preset':'default',
                          #'bitrate':"",
                          'tuning_info': 'high_quality', 
                          'profile': 'high'},
                 device=0):
        """
        Video encoder based on Roman Arzumanyan's VALI. 
        Note: VALI does not implement a muxing step and instead writes a raw encoded file.
        :param output_path: Encoded video file path
        :param width: output video width
        :param height: output video height
        :param codec: encoding codec identifier. Options: 'h264' | 'hvec
        :param fps: output video fps
        :param options; options passed to encoder. may include
            'preset': 'default' | 'P1' | 'P2' | 'P3' | 'P4' | 'P5' | 'P6' | 'P7'
            'bitrate': average encoding bitrate. Can be specified as a number & unit, e.g. '10M' or '1K'
            'tuning_info' : ?? 'high_quality | "ultra_low_latency"'
            'profile': ?? 'high | low '
            'gop': groups of pictures (interval between I-frames). string
            'multipass': "0 | 1"
            "lookahead": "8", # how far to look ahead (more is slower but better quality)
            'bp')
        :param device: what GPU device to use. Defaults to the default GPU (0)
        """
        self.codec = codec
        self.device = device
        self.output_path=output_path
        if codec=='h264':
            maxsize=4096
        elif codec=='hevc':
            maxsize=8192
    
        self.width=min(width, maxsize)
        self.height=min(height, maxsize)
        #self.fps=fps
        #if fps in ffps.keys():
        #    self.fps_num=ffps[fps].numerator
        #    self.fps_den=ffps[fps].denominator
        #else:
        #    self.fps_num=1000 ## *24 if you do not set pkt.time_base
        #    self.fps_den=int(float(self.fps)*1000)
        self._fps=fps2frac(fps)
        self.fps_num=self._fps.numerator
        self.fps_den=self._fps.denominator
        ## vali expects a string for fps...
        fps_str=f"{float(self._fps):.2f}"

        self.backend='vali'
        # Create encoding parameters dictionnary that will be passed to the Nvidia encoder
        enc_params=options
        enc_params.update({'codec':codec,
                            'fps':fps_str,
                            's': f"{str(self.width)}x{str(self.height)}"
                        })
        # Overwrite preset parameters if passed (bugg here!)
        #if 'bitrate' in enc_params.keys() and ( enc_params['bitrate'] is None or enc_params['bitrate'] != "" ):
        #    enc_params['bitrate'] = "10M"

        self.nv_enc = vali.PyNvEncoder(enc_params, self.device)  # VPF encoder
        self.enc_frame = np.ndarray(shape=(0), dtype=np.uint8)  # Encoding buffer

        # Create color converter that converts to the suitable export color space (NV12)
        to_nv12 = ColorConverterVALI(self.width, self.height, self.device)
        to_nv12.add(vali.PixelFormat.RGB, vali.PixelFormat.YUV420)
        to_nv12.add(vali.PixelFormat.YUV420, vali.PixelFormat.NV12)
        self.to_nv12o = to_nv12
        
        ## create uploader and color converters
        self.to_gpu=vali.PyFrameUploader(self.device)

        ## create av muxer
        self.dstFile = av.open(self.output_path, 'w')
        self.out_stream = self.dstFile.add_stream(codec,rate=1)
        self.out_stream.width = self.width
        self.out_stream.height = self.height
        self.framecnt=0
        
        if PYCUDA_GL_AVAILABLE:
            self.pbosize=self.width*self.height*3
            self.pbo=glGenBuffers(1)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, self.pbosize, c_void_p(0),GL_STREAM_DRAW)
            #self.cuda_pbo = RegisteredBuffer(int(self.pbo))
            self.rgb_surface=vali.Surface.Make(vali.PixelFormat.RGB,self.width,self.height, self.device)
            self.to_gpu.Run(np.array([0]*self.width*self.height*3,dtype=np.uint8),self.rgb_surface)

            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
            logging.getLogger().info("Using direct OpenGL to nvenc path")
        else:
            logging.getLogger().info("Could not setup direct path from openGL to hadware encoder")
            logging.getLogger().info("Uning CPU copies instead")

    def putframe(self,data,gpucpy=True):
        if not isinstance(data,int):
            self.rgb_surface = self.to_gpu.UploadSingleFrame(np.array(np.frombuffer(data,dtype=np.uint8)))
        else:
            if not gpucpy or not PYCUDA_GL_AVAILABLE:
                ## read texture data into system memory, then send it back to gpu
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, data)
                glBindBuffer(GL_PIXEL_PACK_BUFFER,0)
                buffer=glGetTexImage(GL_TEXTURE_2D,0,GL_RGB,GL_UNSIGNED_BYTE)
                self.to_gpu.Run(np.array(np.frombuffer(buffer,dtype=np.uint8)),self.rgb_surface)
            else:
                ## read texture data into buffer
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, data)
                glBindBuffer(GL_PIXEL_PACK_BUFFER, int(self.pbo))
                glGetTexImage(GL_TEXTURE_2D,0,GL_RGB,GL_UNSIGNED_BYTE,array=c_void_p(0))
                ## copy to rgb_surface
                cuda_pbo = RegisteredBuffer(int(self.pbo))
                buffer_mapping = cuda_pbo.map()
                buffptr,buffsize=buffer_mapping.device_ptr_and_size()
                cpy = pycuda.driver.Memcpy2D()
                cpy.set_dst_device(self.rgb_surface.Planes[0].GpuMem)
                cpy.set_src_device(buffptr)
                cpy.width_in_bytes = self.width*3 
                cpy.src_pitch = self.width*3
                cpy.dst_pitch = self.rgb_surface.Planes[0].Pitch
                cpy.height = self.rgb_surface.Planes[0].Height
                cpy(aligned=False)
                pycuda.driver.Context.synchronize()
                buffer_mapping.unmap()
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)

        ## convert rgb data in self.rgbsurface to nv12
        ## the rest is just a matter of converting pixels
        rawnv12=self.to_nv12o.run(self.rgb_surface)
        import time
        time.sleep(0.001) ## we must wait for conversion to finish! maybe the bug is solved?
        success = self.nv_enc.EncodeSingleSurface(rawnv12, self.enc_frame, sync = True)
        if success:
            self.mux(self.enc_frame)

    def mux(self,frame):
        encByteArray = bytearray(self.enc_frame)  # encFrame from EncodeSingleSurface
        pkt = av.packet.Packet(self.enc_frame)
        pkt.pts = self.framecnt*self.fps_den
        pkt.dts = self.framecnt*self.fps_den
        pkt.time_base = av.utils.Fraction(1, self.fps_num)
        pkt.stream = self.out_stream  # attach pkt to the stream
        self.dstFile.mux(pkt)
        self.framecnt+=1

    def close(self):
        while True:
            success = self.nv_enc.FlushSinglePacket(self.enc_frame)
            if success:
                self.mux(self.enc_frame)
            else:
                break
        self.dstFile.close()