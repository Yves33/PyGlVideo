import os, pathlib
if not "TEST_VIDEO_FILE" in os.environ:
    print("You must define TEST_VIDEO_FILE environment variable to point to a video file for testing. ")
    exit()
assert(pathlib.Path(os.environ["TEST_VIDEO_FILE"]).exists())

os.environ["IMGUI_IMPL"]="BUNDLE"
import OpenGL
OpenGL.ERROR_ON_COPY = True
from OpenGL.GL import *
import time
import numpy as np
import fractions
import moderngl_window as mglw
if os.environ["IMGUI_IMPL"]!="BUNDLE":
    import imgui
    from moderngl_window.integrations.imgui import ModernglWindowRenderer
    ImTextureRef=lambda x:x
    naked_window_flags=imgui.WINDOW_NO_SCROLLBAR|imgui.WINDOW_NO_COLLAPSE
    text_input_flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
    BUNDLEAPI='bundle' in str(imgui)
else:
    from imgui_bundle import imgui
    from moderngl_window_integrations_imgui_bundle import ModernglWindowRenderer
    ImTextureRef=lambda x:imgui.ImTextureRef(x)
    naked_window_flags=imgui.WindowFlags_.no_scrollbar|imgui.WindowFlags_.no_collapse
    text_input_flags=imgui.InputTextFlags_.enter_returns_true
    BUNDLEAPI='bundle' in str(imgui)

try:
    import pycuda
    PYCUDA_AVAILABLE=True
except:
    PYCUDA_AVAILABLE=False

from pyglvideo import FrameGrabber
from pyglvideo import FrameRecorder
from pyglvideo import (GLBridgeYUV420,   # for gpu->(cpu)->gl->yuv_2_rgb
                     GLBridgeNV12,     # for gpu->(cpu)->gl->nv12_2_rgb
                     GLBridgeRGB,      # for gpu->(cpu)->gl
                     CudaBridgeNV12,   # for avgl->y+uv->cuda->__dlpack__
                     )

## notes
## GLBridgeYUV420 handles both yuv420 and yuv444. could be easily adapted to any planar format (rgb_planar)
## GLBridgeNV12   handles both nv12 and nv24 (not tested). could be easily adapted for nv16
## RGB            handles interleaved rgb. no shader, no brightness / contrast adjustment
settings={
    'dec':{
        'av':{
            ## with hwaccel='cuda' and is_hw_owned=True, av outputs 1 large y+uv plane (w, h*3//2)
            ## with hwaccel='cuda' and is_hw_owned=False, av outputs y and uv planes
            ## with hwaccel=None and is_hw_owned=False, av outputs y, u and v planes
            'hwaccel':'cuda',     ## the difference is not that big, unless is_hw_owned==True. cuda may be a little faster.
            'is_hw_owned':True,   ## keep data to cuda on hw, otherwise is fetched back to cpu
            },
        'vali':{
            ## vali decoder has the option do convert the surface to rgb, yuv, or keep nv12
            'tgt_format':'nv12', ## keep data to cuda on hw, otherwise is fetched back to cpu
            'is_hw_owned':True,
            },
        'pil':{},
        },
    'enc':{
        'vali': {
            'codec':'h264', ## h264 or hevc
            'fps':29.97,    ## note that we provide fps as float and not as str!
            'device':0,
            'options':{
                'preset':'default',
                'bitrate':'10M',
                'tuning_info': 'high_quality', 
                'profile': 'high',
                'gop':'15'
                },
            },
        'av':{
            'fps':29.97,
            ## use h264_nvenc, hevc_nvenc for hardware encode. see ffmpeg -codecs, and same options as ffmpeg cli except for '-' sign
            ## pixel_format refers to destination pixel format, stored in stream. must be nv12 for gl->cuda direct transfer
            ## set gl_convert to perform rgb->nv12 conversion in shader
            ## to get cuda->opengl->cuda, use {'hwaccel':'cuda','is_hw_owned':True}
            'codec':'h264_nvenc','options':{'pixel_format':'nv12','preset':'slow','crf':'22'},'glconvert':True,
            #'codec':'h264_nvenc','options':{'pixel_format':'nv12','preset':'slow','crf':'22'},    'glconvert':False,
            #'codec':'h264',      'options':{'pixel_format':'nv12','preset':'slow','crf':'22'},    'glconvert':True,
            #'codec':'h264',      'options':{'pixel_format':'nv12','preset':'slow','crf':'22'},    'glconvert':False,
            }
        }
}
BACKEND='av'

class FPSCounter:
    def __init__(self,interval):
        self.interval=interval
        self.frame_count=0
        self.seconds=time.time_ns()/1e9
        self._fps=0
    
    def fps(self):
        self.frame_count+=1
        if (self.seconds+self.interval)<time.time_ns()/1e9:
            self._fps=self.frame_count/self.interval
            self.frame_count=0
            self.seconds=time.time_ns()/1e9
        return self._fps

class MovieController:
    def __init__(self,grabber):
        self.last_ticks=-1
        self.grabber=grabber
        self.speed=1
        self.status=0

    def __getattr__(self, attr):
        return getattr(self.grabber, attr)
    
    def play(self):
        self.status=1
        self.last_ticks=time.perf_counter()

    def pause(self):
        self.status=0
        self.last_ticks=0

    def toggle(self):
        self.pause() if self.status else self.play()

    def load(self,filename):
        self.grabber.load(filename)
        self.frame_delta=1/self.grabber.frames_per_sec

    def getframe(self, force=False):
        if force or (self.status==1 and time.perf_counter()>self.last_ticks+float(self.frame_delta)/self.speed):
            self.grabber.readframe()
            self.last_ticks=time.perf_counter()
            return True
        return False
    
    def seek(self,pos=None, frame=None,exact_frame=True):
        ## seeking by frame number is ususally not supported. some backends implement a trick, but ususally recompute internally the timestamps
        # PyAV => no support
        # VALI => supported for non VFR videos
        # in order to be coherent, we rely on finding pts, then seeking
        if not (pos is None):
            self.grabber.seek(pos,exact_frame=True)
        else:
            self.grabber.seek(self.grabber.position_from_index(frame),exact_frame=True)

def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True

class WindowEvents(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "imgui Integration"
    resource_dir = (pathlib.Path(__file__)).parent.resolve()
    aspect_ratio = None
    vsync=False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        imgui.create_context()
        self.wnd.ctx.error
        self.imgui = ModernglWindowRenderer(self.wnd)

        ## we create an fbo that we will display later using imgui.image()
        self.fbo_width,self.fbo_height=854,int(854/16*9) ## 16:9 aspect ratio
        self.fbo = self.ctx.framebuffer(color_attachments=self.ctx.texture((self.fbo_width,self.fbo_height), 3))
        self.imgui.register_texture(self.fbo.color_attachments[0])
        self.program = self.load_program(path=pathlib.Path(__file__).parent/'flat.glsl')

        ## create buffers
        vertices=np.array([[0.0,0.0], [1.0,0.0], [1.0,1.0], [0.0,1.0]],dtype='<f4')*2-1 ## scale to -1,1
        uv=np.array([[0.0,0.0], [1.0,0.0], [1.0,1.0], [0.0,1.0]],dtype='<f4')
        faces=np.array([0,1,2 ,0,2,3 ],dtype='<i2')
        self.vbo = self.ctx.buffer(np.hstack((vertices,uv)).flatten() )
        self.ibo = self.ctx.buffer(faces)
        self.vao = self.ctx.vertex_array(self.program, 
                                        self.vbo, 'a_position', 'a_uv',
                                        index_buffer=self.ibo,
                                        index_element_size=2)
        
        srcfile=os.environ["TEST_VIDEO_FILE"]
        self.loadfile(srcfile,backend=BACKEND)
        self.imgui.register_texture(self.player_texture)
        self.recorder=None
        self.recording=False
        self.fpscounter=FPSCounter(1)
        self.player.speed=1

        self.cliptime=0
        self.last_ticks=-1

        ## we can directly send rgb frames to av that will convert them to appropriate pixel format
        ## we can also perform rgb->nv12 conversion in shader, 
        ## in which case we can transfer data to av either thrrough np array or gpu array
        ## for the demo, we always create the gl rgb_to_nv12 conversion
        ## if glconvert is False: data is forwarded to av as rgb buffer
        ## if glconvert is True:
        ##   if encoder is xx_nvenc: data is forwarded to av through y+uv cuda buffers (__dlpack__())
        ##   otherwise: data is forwarded through y+uv numpy buffers (__dlpack__()) 

        self.cudabridge=CudaBridgeNV12(self.fbo.color_attachments[0].glo,
                                       self.fbo_width,self.fbo_height,
                                       target='cuda' if 'nvenc' in settings['enc']['av']['codec'] else 'cpu')
        ## for debugging rgb2nv12, make sure that we can display the textures in imgui
        self.ytex=self.ctx.external_texture(self.cudabridge.ytex,
                                        (self.cudabridge.width,self.cudabridge.height),
                                        1,0,
                                        'f1')
        self.imgui.register_texture(self.ytex)
        self.uvtex=self.ctx.external_texture(self.cudabridge.uvtex,
                                        (self.cudabridge.width,self.cudabridge.height),
                                        2,0,
                                        'f1')
        self.imgui.register_texture(self.uvtex)

    def loadfile(self,srcfile,backend='av',width=None,height=None):
        global PYCUDA_AVAILABLE,PYCUDA_GL_AVAILABLE
        if pathlib.Path(srcfile).suffix.lower() in ['.jpg','.png']:
            backend='pil'
        ## perform some cleanup - not really required for demo
        try:
            self.bridge.release()
            del self.player
            del self.player_texture
            del self.bridge
        except:
            pass
        
        self.player=MovieController(FrameGrabber(backend=backend,**settings['dec'][BACKEND]))
        self.player.load(srcfile)
        self.player_texture=self.ctx.texture((self.player.width,self.player.height),3,None)
        if PYCUDA_AVAILABLE:
            try:
                import pycuda.autoinit
                import pycuda.gl.autoinit
                PYCUDA_GL_AVAILABLE=True
            except:
                PYCUDA_GL_AVAILABLE=False
        else:
            PYCUDA_GL_AVAILABLE=False
        if self.player.backend in['av','vali']:
            if self.player.grabber.px_format in ['yuv420p','yuv420','yuv444']:
                self.bridge=GLBridgeYUV420(self.player.grabber,self.player_texture.glo,self.player.grabber.px_format)
            elif self.player.grabber.px_format in ['nv12','cuda']:
                self.bridge=GLBridgeNV12(self.player.grabber,self.player_texture.glo)
            elif self.player.grabber.px_format in ['rgb']:
                self.bridge=GLBridgeRGB(self.player.grabber,self.player_texture.glo)
            else:
                print(f"unknown pixel format:{self.player.grabber.px_format}")
                exit()
        elif self.player.backend=='pil':
            self.bridge=GLBridgeRGB(self.player.grabber,self.player_texture.glo)
        self.player.status=2
        self.player.getframe()
        self.bridge.blit()
        self.player.status=0
        self.player.toggle()

    def on_render(self, time: float, frametime: float):
        self.wnd.use()
        self.render_ui()
    
    def render_ui(self):
        imgui.new_frame()
        imgui.begin("Controls")
        imgui.text("FPS : "+str(self.fpscounter._fps))
        seek,seekto_s=imgui.input_text("Jump to time",f"{self.player.grabber.movieposition}",flags=text_input_flags)
        if seek and is_number(seekto_s):
            #seekto=int(seekto*30000/1001)*Fraction(1001,30000) ## can also seek with fractional time!
            self.player.seek(pos=float(seekto_s))
            self.bridge.blit()
        seek,seekto_f=imgui.input_text("Jump to frame",f"{int(self.player.grabber.movieidx)}",flags=text_input_flags)
        if seek and is_number(seekto_f):
            self.player.seek(frame=int(seekto_f))
            self.bridge.blit()
        seek,seekto_p=imgui.input_text("Jump to pts",f"{int(self.player.grabber.frame_info.pts)}",flags=text_input_flags)
        #if seek and is_number(seekto_p):
        #    self.player.seek(pos=int(seekto_p)*self.player.grabber._video.time_base)
        #    self.bridge.blit()
        changed, self.recording=imgui.checkbox("Record output",self.recording)
        if changed:
            if self.recording:
                self.recorder=FrameRecorder(f"./output_{self.player.backend}.mp4",
                                            self.fbo_width,self.fbo_height,
                                            backend=self.player.backend,
                                            codec=settings['enc'][BACKEND]['codec'],
                                            options=settings['enc'][BACKEND]['options'])
            else:
                self.recorder.close()
                self.recorder=None
        if imgui.button("Pause" if self.player.status else "Play"):
            self.player.toggle()
        if imgui.button("step"):
            self.player.getframe(force=True)
            self.bridge.blit()
        changed,self.player.speed=imgui.input_float("Speed", self.player.speed)
        imgui.separator
        imgui.text(f"Stream:")
        imgui.text(f" - duration  :{self.player.grabber.movieduration}")
        imgui.text(f" - timebase  :{self.player.grabber.frames_per_sec.numerator}")
        imgui.text(f" - fps  :{self.player.grabber.frames_per_sec.numerator}/{self.player.grabber.frames_per_sec.denominator}")
        imgui.text(f" - first pts :{self.player.grabber.first_pts}")
        imgui.text(f"Frame:")
        imgui.text(f" - timebase:  {self.player.grabber.frame_info.time_base.denominator}")
        imgui.text(f" - pts     :{self.player.grabber.frame_info.pts}")
        imgui.text(f" - index   :{self.player.grabber.movieidx}")
        imgui.text(f" - time    :{self.player.grabber.movieposition}")
        imgui.text("BUNDLE API" if BUNDLEAPI else "CLASSIC API")
        imgui.text(f"Backend : {self.player.backend}")
        imgui.end()
        
        imgui.begin(f"Decoding RGB buffer", flags=naked_window_flags)
        if BUNDLEAPI:
            width=imgui.get_content_region_avail()[0]
            imgui.image(
                        ImTextureRef(self.fbo.color_attachments[0].glo),
                        imgui.ImVec2(width,width*self.fbo_height/self.fbo_width),
                        uv0=imgui.ImVec2(0,0),uv1=imgui.ImVec2(1,1))
        else:
            width=imgui.get_content_region_available()[0]
            imgui.image(self.fbo.color_attachments[0].glo,
                        width,width*self.fbo_height/self.fbo_width,
                        uv0=(0,0),uv1=(1,1))
        imgui.end()
        
        newframe=False
        ## normal way of proceeding
        if 1:
            if self.player.getframe():
                fps=self.fpscounter.fps()
                self.bridge.blit()
                newframe=True
        else:
            ## alternate way using pseudo seek
            ## because "seek" method only seeks when there is a negative or a large positive gap
            ## we can use seek method while keeping the same performance
            perf_counter_=time.perf_counter()
            if perf_counter_-self.last_ticks>fractions.Fraction(1001,30000):
                self.last_ticks=perf_counter_
                self.cliptime+=fractions.Fraction(1001,30000)
                self.player.grabber.seek(self.cliptime)
                fps=self.fpscounter.fps()
                self.bridge.blit()

        ## FBO RENDERING - the program records the fbo, not the decoded texture nore the screen!
        self.fbo.use()
        self.fbo.clear(0.0,0.0,0.0)
        self.player_texture.use(0)
        self.program['u_tex0']=0
        self.vao.render(6)
        self.wnd.use()
        
        ## theoretically, we only need this if we want the gl to perform rgb->nv12 conversion
        self.cudabridge.blit() ## modifies the current viewport!
        imgui.begin("Encoding - uv buffer")
        width=imgui.get_content_region_avail()[0]
        imgui.image(ImTextureRef(self.uvtex.glo),
                    imgui.ImVec2(width,width*self.cudabridge.height/self.cudabridge.width),
                    uv0=imgui.ImVec2(0,0),uv1=imgui.ImVec2(1.0,1.0))
        imgui.end()
        ##

        if newframe and self.recorder:
            if BACKEND=='av' and settings['enc'][BACKEND]['glconvert']:
                self.recorder.putframe((self.cudabridge.y_plane,self.cudabridge.uv_plane))
            else:
                self.recorder.putframe(int(self.fbo.color_attachments[0].glo),gpucpy=PYCUDA_AVAILABLE)
               
        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def on_resize(self, width: int, height: int):
        #self.prog["m_proj"].write(glm.perspective(glm.radians(75), self.wnd.aspect_ratio, 1, 100))
        self.imgui.resize(width, height)

    def on_key_event(self, key, action, modifiers):
        self.imgui.key_event(key, action, modifiers)

    def on_mouse_position_event(self, x, y, dx, dy):
        self.imgui.mouse_position_event(x, y, dx, dy)

    def on_mouse_drag_event(self, x, y, dx, dy):
        self.imgui.mouse_drag_event(x, y, dx, dy)

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.imgui.mouse_scroll_event(x_offset, y_offset)

    def on_mouse_press_event(self, x, y, button):
        self.imgui.mouse_press_event(x, y, button)

    def on_mouse_release_event(self, x: int, y: int, button: int):
        self.imgui.mouse_release_event(x, y, button)

    def on_unicode_char_entered(self, char):
        self.imgui.unicode_char_entered(char)


if __name__ == "__main__":
    mglw.run_window_config(WindowEvents)
