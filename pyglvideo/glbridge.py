from OpenGL.GL import *
from ctypes import c_void_p,c_byte,POINTER,memmove,cast
from collections import namedtuple
import numpy as np
import itertools

try:
    import pycuda
    from pycuda.gl import RegisteredBuffer
    from pycuda.gpuarray import GPUArray ## requires pytools
    import dlpack
    ##from dlpack import asdlpack,todict
except:
    pass

## for GPU blitter compatibility with VALI, we need to import ctypes and dlpack
import ctypes
ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


## wget https://github.com/dmlc/dlpack/blob/main/apps/numpy_dlpack/dlpack/dlpack.py
## can be replaced by https://github.com/pearu/pydlpack/blob/main/dlpack/__init__.py, which offers bi directionnal communication!
#from .dlpack import _c_str_dltensor, DLManagedTensor

ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

gl_unpack_info=namedtuple('gl_unpack_info',['height',
                                            'width',
                                            'stride',
                                            'components',
                                            'width_in_bytes',
                                            'size_in_bytes',
                                            'ptr',
                                            'device_id',
                                            'device_type',
                                            'typestr'])
def __gl_unpack__(plane):
    try:
        info=dlpack.todict(plane.__dlpack__())['dl_tensor']
    except:
        ## when av exports planar format, it outputs an unidimensionnal buffer
        ## unfortunately, reshaping copies the buffer!
        ## pycapsule=np.array(plane).reshape(plane.height,plane.width).__dlpack__()
        ## but we can re-create the np.array using ctypes
        ## the code assumes that plane.__dlpack__() never fails when data is on gpu!
        ## because stride may be present, we cannot directly use ctypeslib.asarray().__dlpack__
        #info=dlpack.todict(np.ctypeslib.as_array(cast(plane.buffer_ptr,POINTER(c_byte)), 
        #                                shape=(plane.height, plane.width)).__dlpack__())['dl_tensor']
        _buf=np.ctypeslib.as_array(cast(plane.buffer_ptr,POINTER(c_byte)), 
                                        shape=(1,plane.buffer_size))
        info=dlpack.todict(np.lib.stride_tricks.as_strided(_buf,
                                                           shape=(plane.height, plane.width),
                                                           strides=(plane.line_size, 1),
                                                           ).__dlpack__())['dl_tensor']
        
    height,width=info['shape'][0],info['shape'][1]
    stride=info['strides'][0] if info['strides'] else list(itertools.accumulate(info['shape'][1:],lambda a,b:a*b))[-1]
    n_components=info['shape'][2] if len(info['shape'])>2 else stride//width
    # if True or len(info['shape'])<3:
    #     stride=info['strides'][0] if info['strides'] else width
    #     n_components=stride//width  # info['shape'][2] if len(info['shape])>2 else 1
    # else: # numpy case
    #     n_components=info['shape'][2]
    #     stride=width*n_components
    width_in_bytes=stride       #*n_components
    size_in_bytes=width_in_bytes*height
    ptr=info['data']
    device_id=info['device']['device_id']
    device_type=2 if info['device']['device_type']=='DLCUDA' else 1
    typestr=info['dtype']
    return gl_unpack_info(height=height,
                          width=width,
                          stride=stride,
                          components=n_components,
                          width_in_bytes=width_in_bytes,
                          size_in_bytes=size_in_bytes,
                          ptr=ptr,
                          device_id=device_id,
                          device_type=device_type,
                          typestr=None ## don't need it
                          )
    
## refactoring suggestions
# class cuda_gl_buffer
#   def __init__(self,size:int):
#   def copy_from_from_dlpack(self,arr): fills buffer with content of arr
#   @property
#   def glo(self):returns the gl buffer identifier (or gl_id) 

class GLBridgeOneshot:
    ## one shot blitter for still images
    def __init__(self,player,player_texture):
        self.player=player
        self.player_texture=player_texture
        self.fired=True
        self.brightness=0.0
        self.contrast=1.0

    def release(self):
        pass

    def blit(self,unit=GL_TEXTURE0):
        if self.fired:
            return
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D,self.player_texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0,0, 
                        self.player.width, self.player.height,
                        GL_RGB, GL_UNSIGNED_BYTE, 
                        self.player.rgb.ctypes.data_as(POINTER(c_byte)) ## tobytes() will perform a copy of the array!
                )
        self.fired=True

class CudaBridgeNV12:
    def __init__(self,src_texture,width=None, height=None,px_format='nv12',target='cuda'):
        '''create_cuda_buffers should only be set to true when the
        gl_only: do not create buffer. just make rgb->nv12 conversion
        cpu: put y and uv planes in numpy arrays (otherwise cuda)
        '''
        self.src_texture=src_texture
        if width is None or height is None:
            self.width=glGetTexLevelParameteriv(self.src_texture,0,GL_TEXTURE_WIDTH)
            self.height=glGetTexLevelParameteriv(self.src_texture,0,GL_TEXTURE_HEIGHT)
        else:
            self.width=width
            self.height=height
        self.px_format=px_format
        self.uv_scale=2 if self.px_format=='nv12' else 1 ## to be adjusted. one needs 1 for nv24 and x/y secific for nv16
        self.mkshaders()
        self.mktextures()
        ## the bridge may be used only to perform rgb->y+uv transfer, while keeping everything to the gl side!
        self.target=target ## if None or False,buffers wil not be created
        if self.target:
            self.mkbuffers()

    def __del__(self):
        try:
            self.release()
        except:
            pass

    def release(self):
        ## must be explicilety called. we cannot rely on __del__ as grabage collection being asynchronous, 
        ## the __del__ function may be called after the context has been destroyed
        glUseProgram(0)
        glDeleteProgram(self.program_y)               # release program
        glDeleteProgram(self.program_uv) 
        for t in [self.ytex,self.uvtex]:   # release all threee textures
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D,0)
            if glIsTexture(t):
                glDeleteTextures(t)
        glBindFramebuffer(GL_FRAMEBUFFER,0)
        glDeleteFramebuffers(2,np.array([self.fbo_y,self.fbo_uv]))
        if self.target:
            glDeleteBuffers(2,np.array([self.pbo_y,self.pbo_uv]))                   # release pbo
        if self.target=='cuda':
            del self.cuda_y
            del self.cuda_uv
        elif self.target=='cpu':
            del self.cpu_y
            del self.cpu_uv

    def mkshaders(self):
        cvt='''
        y= 0.257*r + 0.504*g +0.098*b;
        v= 0.439*r - 0.368*g - 0.071*b + 0.5;
        u=-0.148*r - 0.291*g + 0.439*b + 0.5;
        '''
        vertex_shader_source = """
            #version 330 core
            in vec3 a_position;
            in vec2 a_uv;
            out vec2 uv;
            void main( void)
            {
            // Declare a hard-coded array of positions
            const vec2 vertices[6] = vec2[6](vec2(-0.5, 0.5),
            vec2( 0.5, 0.5),
            vec2( 0.5, -0.5),
            vec2(-0.5, 0.5),
            vec2( 0.5, -0.5),
            vec2(-0.5, -0.5));
            // Index into our array using gl_VertexID
            uv=vertices[gl_VertexID]+vec2(0.5,0.5);
            gl_Position = vec4(2*vertices[gl_VertexID],1.0,1.0);
            }
            """
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, vertex_shader_source)
        glCompileShader(vertex_shader)

        fragment_shader_source_uv = """
            #version 330 core
            uniform sampler2D rgbtex;
            in vec2 uv;
            out vec4 color;
            void main(void)
            {
                float y, u, v;
                vec4 rgba=texture(rgbtex, uv);
                //y= 0.257*rgba.r + 0.504*rgba.g +0.098*rgba.b;
                v= 0.439*rgba.r - 0.368*rgba.g - 0.071*rgba.b + 0.5;
                u=-0.148*rgba.r - 0.291*rgba.g + 0.439*rgba.b + 0.5;
                //y=clamp(y,0.0,1.0);
                u=clamp(u,0.0,1.0);
                v=clamp(v,0.0,1.0);
                color = vec4(u,v,0.0,1.0);
            }
            """
 
        fragment_shader_uv = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader_uv, fragment_shader_source_uv)
        glCompileShader(fragment_shader_uv)

        self.program_uv = glCreateProgram()
        glAttachShader(self.program_uv, vertex_shader)
        glAttachShader(self.program_uv, fragment_shader_uv)
        glLinkProgram(self.program_uv)
        self.tex_loc_uv=glGetUniformLocation(self.program_uv, b'rgbtex')
        #y= 0.257*r + 0.504*g +0.098*b;
        #v= 0.439*r - 0.368*g - 0.071*b + 0.5;
        #u=-0.148*r - 0.291*g + 0.439*b + 0.5;
        fragment_shader_source_y = """
            #version 330 core
            uniform sampler2D rgbtex;
            in vec2 uv;
            out vec4 color;
            void main(void)
            {
                float y, u, v;
                vec4 rgba=texture(rgbtex, uv);
                y= 0.257*rgba.r + 0.504*rgba.g +0.098*rgba.b;
                //v= 0.439*rgba.r - 0.368*rgba.g - 0.071*rgba.b + 0.5;
                //u=-0.148*rgba.r - 0.291*rgba.g + 0.439*rgba.b + 0.5;
                y=clamp(y,0.0,1.0);
                //u=clamp(u,0.0,1.0);
                //v=clamp(v,0.0,1.0);
                color = vec4(y,0.0,0.0,1.0);
            }
            """
 
        fragment_shader_y = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader_y, fragment_shader_source_y)
        glCompileShader(fragment_shader_y)

        self.program_y = glCreateProgram()
        glAttachShader(self.program_y, vertex_shader)
        glAttachShader(self.program_y, fragment_shader_y)
        glLinkProgram(self.program_y)
        self.tex_loc_y=glGetUniformLocation(self.program_y, b'rgbtex')
        
        # --- Clean up now that we don't need these shaders anymore.
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader_uv)
        glDeleteShader(fragment_shader_y)

    def mktextures(self):
        ##create a texture to hold y and uv components
        self.ytex=glGenTextures(1)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D,self.ytex)
        glTexImage2D(GL_TEXTURE_2D, 
            0, 
            GL_RED, 
            self.width, 
            self.height, 
            0, 
            GL_RED, 
            GL_UNSIGNED_BYTE, 
            None
            )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D,0)

        self.uvtex=glGenTextures(1)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D,self.uvtex)
        ## the size of the uv texture depends on pixel format
        ## nv12 width//2, height//2
        ## nv16 width//2, height
        ## nv24 width, height
        ## we assume we only get nv12 pixels (to be improved, obviously)
        glTexImage2D(GL_TEXTURE_2D, 
            0, 
            GL_RG, 
            self.width//self.uv_scale,
            self.height//self.uv_scale, 
            0, 
            GL_RG, 
            GL_UNSIGNED_BYTE, 
            None
            )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D,0)

        ## create a framebuffer and attach player texture as color attachment
        self.fbo_y=glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER,self.fbo_y)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,self.ytex)
        glFramebufferTexture2D(GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            self.ytex,
            0)
        glBindTexture(GL_TEXTURE_2D,0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        ## create a framebuffer and attach player texture as color attachment
        self.fbo_uv=glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER,self.fbo_uv)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,self.uvtex)
        glFramebufferTexture2D(GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            self.uvtex,
            0)
        glBindTexture(GL_TEXTURE_2D,0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
        ## even if we are rendering without a vao, we must bind an empty vao
        self.vao=glGenVertexArrays(1)
        
    def mkbuffers(self):
        ## create PBO
        ## pbo size also depends on pixel format
        ## nv12 3*(self.player.width*self.player.height)//2
        ## nv16 4*(self.player.width*self.player.height)//2
        ## nv24 6*(self.player.width*self.player.height)//2
        ## moreover, due to data alignment, we must rely on pitch rather than width
        ## does not really matter here, as the pbo is orphaned before each blit
        self.pbosize_y=self.width*self.height
        self.pbo_y=GLuint(0);glGenBuffers(1,self.pbo_y);self.pbo_y=self.pbo_y.value ## hack to make it work with pyglet
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER,self.pbo_y)
        glBufferData(GL_PIXEL_UNPACK_BUFFER,self.pbosize_y,c_void_p(0),GL_STREAM_READ)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        if self.target=='cuda':
            self.cuda_y=GPUArray((self.height,self.width),np.uint8)
        elif self.target=='cpu':
            self.cpu_y=np.empty((self.height, self.width),dtype=np.uint8)
        
        self.pbosize_uv=self.width*self.height//2
        self.pbo_uv=GLuint(0);glGenBuffers(1,self.pbo_uv);self.pbo_uv=self.pbo_uv.value ## hack to make it work with pyglet
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER,self.pbo_uv)
        glBufferData(GL_PIXEL_UNPACK_BUFFER,self.pbosize_uv,c_void_p(0),GL_STREAM_READ)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        if self.target=='cuda':
            self.cuda_uv=GPUArray((self.height//2,self.width,),np.uint8)
        elif self.target=='cpu':
            self.cpu_uv=np.empty((self.height//2, self.width),dtype=np.uint8)

    def blit(self):
        ## draw to our framebuffer, ie player texture
        vp=glGetIntegerv(GL_VIEWPORT)             ## the penalty introduced by this glGet is minimal
        glBindFramebuffer(GL_FRAMEBUFFER,self.fbo_y)
        glUseProgram(self.program_y)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.src_texture)
        glUniform1i(self.tex_loc_y,0)
        glBindVertexArray(self.vao)
        glViewport(0, 0, self.width, self.height)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindVertexArray(0)
        if self.target:
            glBindBuffer(GL_PIXEL_PACK_BUFFER, self.pbo_y)
            glBufferData(GL_PIXEL_PACK_BUFFER, self.pbosize_y,None, GL_STREAM_READ)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.ytex)
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RED,GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        glBindFramebuffer(GL_FRAMEBUFFER,0)

        ## draw to our framebuffer, ie player texture
        glBindFramebuffer(GL_FRAMEBUFFER,self.fbo_uv)
        glUseProgram(self.program_uv)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.src_texture)
        glUniform1i(self.tex_loc_uv,0)
        glBindVertexArray(self.vao)
        glViewport(0, 0, self.width//2, self.height//2)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindVertexArray(0)
        if self.target:
            glBindBuffer(GL_PIXEL_PACK_BUFFER, self.pbo_uv)
            glBufferData(GL_PIXEL_PACK_BUFFER, self.pbosize_uv,None, GL_STREAM_READ)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.uvtex)
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RG,GL_UNSIGNED_BYTE,ctypes.c_void_p(0))
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        glBindFramebuffer(GL_FRAMEBUFFER,0)

        ## transfer to buffers, either numpy or cuda!
        if self.target=='cpu':
            glBindBuffer(GL_PIXEL_PACK_BUFFER, self.pbo_y)
            glGetBufferSubData(GL_PIXEL_PACK_BUFFER,0,self.width*self.height,
                               ctypes.c_void_p(self.cpu_y.ctypes.data))
            glBindBuffer(GL_PIXEL_PACK_BUFFER, self.pbo_uv)
            glGetBufferSubData(GL_PIXEL_PACK_BUFFER,0,self.width*self.height//2,
                                ctypes.c_void_p(self.cpu_uv.ctypes.data))
            ## from PIL import Image
            ## Image.fromarray(self.cpu_uv).show()
        elif self.target=='cuda':
            cuda_pbo = RegisteredBuffer(self.pbo_y)           ## in order to get gpu<->gpu copies in both encoder and decoder, one need to re-register the buffer!
            buffer_mapping = cuda_pbo.map()
            buffptr,buffsize=buffer_mapping.device_ptr_and_size()
            dst_offset=0 ## offset in gl buffer
            cpy = pycuda.driver.Memcpy2D()                      ## don't need to recreate it each frame?
            cpy.set_src_device(buffptr) ##(info.ptr)
            cpy.set_dst_device(self.cuda_y.ptr)        ##(buffptr+dst_offset)                       
            cpy.width_in_bytes = self.width            ##info.width_in_bytes ##pbuff.width*pbuff.components
            cpy.src_pitch = 0                          #info.stride    ## plane.pitch
            cpy.dst_pitch = self.cuda_y.strides[0]     ##info.stride    ## because glbuffer is mapped to cuda, it uses the same stride; we later specify glPixelStorei(GL_UNPACK_ROW_LENGTH, self.player.decoded.planes[0].line_size)
            cpy.height = self.height       ## plane.height
            cpy(aligned=False)
            pycuda.driver.Context.synchronize()
            buffer_mapping.unmap()
            #from PIL import Image
            #Image.fromarray(self.cuda_y.get()).show()
            cuda_pbo = RegisteredBuffer(self.pbo_uv)           ## in order to get gpu<->gpu copies in both encoder and decoder, one need to re-register the buffer!
            buffer_mapping = cuda_pbo.map()
            buffptr,buffsize=buffer_mapping.device_ptr_and_size()
            dst_offset=0 ## offset in gl buffer
            cpy = pycuda.driver.Memcpy2D()
            cpy.set_src_device(buffptr)
            cpy.set_dst_device(self.cuda_uv.ptr)
            cpy.width_in_bytes = self.width
            cpy.src_pitch = 0
            cpy.dst_pitch = self.cuda_uv.strides[0]
            cpy.height = self.height//2
            cpy(aligned=False)
            pycuda.driver.Context.synchronize()
            buffer_mapping.unmap()

        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        glUseProgram(0)
        glViewport(*vp)

    @property
    def y_plane(self):
        if self.target=='cpu':
            return self.cpu_y
        elif self.target=='cuda':
            return dlpack.asdlpack(self.cuda_y)
    
    @property
    def uv_plane(self):
        if self.target=='cpu':
            return self.cpu_uv
        elif self.target=='cuda':
            return dlpack.asdlpack(self.cuda_uv)

class GLBridgeNV12:
    def __init__(self,player,player_texture,px_format='nv12',dual=False):
        self.player=player
        self.player_texture=player_texture
        self.dual=dual
        self.px_format=px_format
        self.uv_scale=2 if self.px_format=='nv12' else 1 ## to be adjusted. one needs 1 for nv24 and x/y secific for nv16
        self.mkshaders()
        self.mkbuffers()
        if self.dual:
            self.pbos=itertools.cycle([self.pbo1,self.pbo2])
        else:
            self.pbos=itertools.cycle([self.pbo1])
        self.contrast=1.0
        self.brightness=0.0

    def __del__(self):
        try:
            self.release()
        except:
            pass

    def release(self):
        ## must be explicilety called. we cannot rely on __del__ as grabage collection being asynchronous, 
        ## the __del__ function may be called after the context has been destroyed
        glUseProgram(0)
        glDeleteProgram(self.program)               # release program
        for t in [self.ytex,self.utex,self.vtex]:   # release all threee textures
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D,0)
            if glIsTexture(t):
                glDeleteTextures(t)
        glBindFramebuffer(GL_FRAMEBUFFER,0)
        glDeleteFramebuffers(1,[self.fbo])
        glDeleteBuffers(1,[self.pbo1])                   # release pbo
        if self.dual:
            glDeleteBuffers(1,[self.pbo2]) 

    def mkshaders(self):
        vertex_shader_source = """
            #version 330 core
            in vec3 a_position;
            in vec2 a_uv;
            out vec2 uv;
            void main( void)
            {
            // Declare a hard-coded array of positions
            const vec2 vertices[6] = vec2[6](vec2(-0.5, 0.5),
                                             vec2( 0.5, 0.5),
                                             vec2( 0.5, -0.5),
                                             vec2(-0.5, 0.5),
                                             vec2( 0.5, -0.5),
                                             vec2(-0.5, -0.5));
            // Index into our array using gl_VertexID
            uv=vertices[gl_VertexID]+vec2(0.5,0.5);
            gl_Position = vec4(2*vertices[gl_VertexID],1.0,1.0);
            }
            """
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, vertex_shader_source)
        glCompileShader(vertex_shader)

        fragment_shader_source = """
            #version 330 core
            uniform sampler2D ytex;
            uniform sampler2D utex;
            uniform float sContrastValue;
            uniform float sBrightnessValue;
            /*uniform sampler2D vtex;*/
            in vec2 uv;
            out vec4 color;
            void main(void)
            {
                float r, g, b, y, u, v;
                y=texture(ytex, uv).r;
                u=texture(utex, uv).r-0.5;
                v=texture(utex, uv).g-0.5;
                /* not faster!
                vec3 rgb= mat3(    1,       1,     1,
                                   0, -.34413, 1.772,
                               1.402, -.71414,     0)*vec3(y,u,v);
                color=vec4(rgb,1.0);
                */
                r = y + 1.13983*v;
                g = y - 0.39465*u - 0.58060*v;
                b = y + 2.03211*u;
                r = r * sContrastValue + sBrightnessValue;
                g = g * sContrastValue + sBrightnessValue;
                b = b * sContrastValue + sBrightnessValue;
                r = clamp(r, 0.0, 1.0);
                g = clamp(g, 0.0, 1.0);
                b = clamp(b, 0.0, 1.0);
                color = vec4(r,g,b,1.0);
            }
            """
 
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, fragment_shader_source)
        glCompileShader(fragment_shader)

        self.program = glCreateProgram()
        glAttachShader(self.program, vertex_shader)
        glAttachShader(self.program, fragment_shader)
        glLinkProgram(self.program)
        # --- Clean up now that we don't need these shaders anymore.
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        
        self.ytex_loc=glGetUniformLocation(self.program, b'ytex')
        self.utex_loc=glGetUniformLocation(self.program, b'utex')
        self.contrast_loc=glGetUniformLocation(self.program, b'sContrastValue')
        self.brightness_loc=glGetUniformLocation(self.program, b'sBrightnessValue')
        #self.vtex_loc=glGetUniformLocation(self.program, b'vtex')

    def mkbuffers(self):
        ##create a texture to hold y and uv components
        self.ytex=glGenTextures(1)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D,self.ytex)
        glTexImage2D(GL_TEXTURE_2D, 
            0, 
            GL_RED, 
            self.player.width, 
            self.player.height, 
            0, 
            GL_RED, 
            GL_UNSIGNED_BYTE, 
            None
            )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D,0)

        self.utex=glGenTextures(1)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D,self.utex)
        ## the size of the uv texture depends on pixel format
        ## nv12 width//2, height//2
        ## nv16 width//2, height
        ## nv24 width, height
        ## we assume we only get nv12 pixels (to be improved, obviously)
        glTexImage2D(GL_TEXTURE_2D, 
            0, 
            GL_RG, 
            self.player.width//self.uv_scale,
            self.player.height//self.uv_scale, 
            0, 
            GL_RG, 
            GL_UNSIGNED_BYTE, 
            None
            )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D,0)

        ## create a framebuffer and attach player texture as color attachment
        self.fbo=glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER,self.fbo)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,self.player_texture)
        glFramebufferTexture2D(GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            self.player_texture,
            0)
        glBindTexture(GL_TEXTURE_2D,0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        ## create PBO
        ## pbo size also depends on pixel format
        ## nv12 3*(self.player.width*self.player.height)//2
        ## nv16 4*(self.player.width*self.player.height)//2
        ## nv24 6*(self.player.width*self.player.height)//2
        ## moreover, due to data alignment, we must rely on pitch rather than width
        ## does not really matter here, as the pbo is orphaned before each blit
        self.pbosize=3*self.player.width*self.player.height//2
        self.pbo1=GLuint(0);glGenBuffers(1,self.pbo1);self.pbo1=self.pbo1.value ## hack to make it work with pyglet
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER,self.pbo1)
        glBufferData(GL_PIXEL_UNPACK_BUFFER,self.pbosize,c_void_p(0),GL_STREAM_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        if self.dual:
            self.pbo2=GLuint(0);glGenBuffers(1,self.pbo2);self.pbo2=self.pbo2.value ## hack to make it work with pyglet
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER,self.pbo2)
            glBufferData(GL_PIXEL_UNPACK_BUFFER,self.pbosize,c_void_p(0),GL_STREAM_DRAW)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        ## even if we are rendering without a vao, we must bind an empty vao
        self.vao=glGenVertexArrays(1)

    def blit(self,unit=GL_TEXTURE0):
        infos=[__gl_unpack__(p) for p in self.player.planes]
        self.pbosize=np.sum([nfo.size_in_bytes for nfo in infos])
        pbo_=next(self.pbos)
        ##orphaning buffer is often cheaper!
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER,pbo_)
        glBufferData(GL_PIXEL_UNPACK_BUFFER,self.pbosize,c_void_p(0),GL_STREAM_DRAW)

        if all([nfo.device_type==1 for nfo in infos]):  ## data is on cpu
            dst_offset=0
            for info in infos:
                glBufferSubData(GL_PIXEL_UNPACK_BUFFER,
                            dst_offset,  
                            info.width_in_bytes*info.height,
                            cast(info.ptr,POINTER(c_byte))
                            )
                dst_offset+=info.size_in_bytes

        elif all([nfo.device_type==2 for nfo in infos]): ## data is on cuda device
            cuda_pbo = RegisteredBuffer(pbo_)           ## in order to get gpu<->gpu copies in both encoder and decoder, one need to re-register the buffer!
            buffer_mapping = cuda_pbo.map()
            buffptr,buffsize=buffer_mapping.device_ptr_and_size()
            dst_offset=0 ## offset in gl buffer
            for info in infos:
                cpy = pycuda.driver.Memcpy2D()                      ## don't need to recreate it each frame?
                cpy.set_src_device(info.ptr)
                cpy.set_dst_device(buffptr+dst_offset)                       
                cpy.width_in_bytes = info.width_in_bytes ##pbuff.width*pbuff.components
                cpy.src_pitch = info.stride    ## plane.pitch
                cpy.dst_pitch = info.stride    ## because glbuffer is mapped to cuda, it uses the same stride; we later specify glPixelStorei(GL_UNPACK_ROW_LENGTH, self.player.decoded.planes[0].line_size)
                cpy.height = info.height       ## plane.height
                dst_offset+=info.size_in_bytes ##(pbuff.pitch*pbuff.components)*pbuff.height
                cpy(aligned=False)
            pycuda.driver.Context.synchronize()
            buffer_mapping.unmap()
        
        #update ytex
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,self.ytex)
        glPixelStorei(GL_UNPACK_ROW_LENGTH, infos[0].stride)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0,0, 
            self.player.width, 
            self.player.height,
            GL_RED, 
            GL_UNSIGNED_BYTE, 
            POINTER(c_byte)()
        )
        #glPixelStorei(GL_UNPACK_ROW_LENGTH, 0) ## only required after last transfer!
        #update utex
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D,self.utex)
        glPixelStorei(GL_UNPACK_ROW_LENGTH, infos[0].stride//self.uv_scale)
        ## because vali sends yuv in one plane, we cannot rely on infos[0].height
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 
            self.player.width//self.uv_scale, 
            self.player.height//self.uv_scale,
            GL_RG, 
            GL_UNSIGNED_BYTE, 
            cast(infos[0].width_in_bytes*self.player.height,POINTER(c_byte))
        )
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        ## draw to our framebuffer, ie player texture
        glBindFramebuffer(GL_FRAMEBUFFER,self.fbo)
        glUseProgram(self.program)
        glUniform1i(self.ytex_loc,0)
        glUniform1i(self.utex_loc,1)
        glUniform1f(self.contrast_loc,self.contrast)
        glUniform1f(self.brightness_loc,self.brightness)
        glBindVertexArray(self.vao)
        glViewport(0, 0, self.player.width, self.player.height)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glBindFramebuffer(GL_FRAMEBUFFER,0)

class GLBridgeYUV420:
    def __init__(self,player,player_texture,px_format='yuv420',dual=False):
        self.player=player
        self.player_texture=player_texture
        self.dual=dual
        self.px_format=px_format
        self.uv_scale=2 if self.px_format!='yuv444' else 1 ## 1 for rgb_planar
        self.mkshaders()
        self.mkbuffers()
        if self.dual:
            self.pbos=itertools.cycle([self.pbo1,self.pbo2])
        else:
            self.pbos=itertools.cycle([self.pbo1])
        self.contrast=1.0
        self.brightness=0.0

    def __del__(self):
        try:
            self.release()
        except:
            pass

    def release(self):
        ## must be explicilety called. we cannot rely on __del__ as grabage collection being asynchronous, 
        ## the __del__ function may be called after the context has been destroyed
        glUseProgram(0)
        glDeleteProgram(self.program)               # release program
        for t in [self.ytex,self.utex,self.vtex]:   # release all threee textures
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D,0)
            if glIsTexture(t):
                glDeleteTextures(t)
        glBindFramebuffer(GL_FRAMEBUFFER,0)
        glDeleteFramebuffers(1,[self.fbo])
        glDeleteBuffers(1,[self.pbo1])                   # release pbo
        if self.dual:
            glDeleteBuffers(1,[self.pbo2]) 


    def mkshaders(self):
        vertex_shader_source = """
            #version 330 core
            in vec3 a_position;
            in vec2 a_uv;
            out vec2 uv;
            void main( void)
            {
            // Declare a hard-coded array of positions
            const vec2 vertices[6] = vec2[6](vec2(-0.5, 0.5),
            vec2( 0.5, 0.5),
            vec2( 0.5, -0.5),
            vec2(-0.5, 0.5),
            vec2( 0.5, -0.5),
            vec2(-0.5, -0.5));
            // Index into our array using gl_VertexID
            uv=vertices[gl_VertexID]+vec2(0.5,0.5);
            gl_Position = vec4(2*vertices[gl_VertexID],1.0,1.0);
            }
            """
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, vertex_shader_source)
        glCompileShader(vertex_shader)

        fragment_shader_source = """
            #version 330 core
            uniform sampler2D ytex;
            uniform sampler2D utex;
            uniform sampler2D vtex;
            uniform float sContrastValue;
            uniform float sBrightnessValue;
            in vec2 uv;
            out vec4 color;
            void main(void)
            {
                float r, g, b, y, u, v;
                y=texture(ytex, uv).r;
                u=texture(utex, uv).r-0.5;
                v=texture(vtex, uv).r-0.5;
                /* not faster!
                vec3 rgb= mat3(    1,       1,     1,
                                   0, -.34413, 1.772,
                               1.402, -.71414,     0)*vec3(y,u,v);
                color=vec4(rgb,1.0);
                */
                r = y + 1.13983*v;
                g = y - 0.39465*u - 0.58060*v;
                b = y + 2.03211*u;
                r = r * sContrastValue + sBrightnessValue;
                g = g * sContrastValue + sBrightnessValue;
                b = b * sContrastValue + sBrightnessValue;
                r = clamp(r, 0.0, 1.0);
                g = clamp(g, 0.0, 1.0);
                b = clamp(b, 0.0, 1.0);
                color = vec4(r,g,b,1.0);
            }
            """
 
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, fragment_shader_source)
        glCompileShader(fragment_shader)

        self.program = glCreateProgram()
        glAttachShader(self.program, vertex_shader)
        glAttachShader(self.program, fragment_shader)
        glLinkProgram(self.program)
        # --- Clean up now that we don't need these shaders anymore.
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        
        self.ytex_loc=glGetUniformLocation(self.program, b'ytex')
        self.utex_loc=glGetUniformLocation(self.program, b'utex')
        self.vtex_loc=glGetUniformLocation(self.program, b'vtex')
        self.contrast_loc=glGetUniformLocation(self.program, b'sContrastValue')
        self.brightness_loc=glGetUniformLocation(self.program, b'sBrightnessValue')

    def mkbuffers(self):
        ##create a texture to hold ytexinance //chrominance //vtexuration
        self.ytex=glGenTextures(1)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D,self.ytex)
        glTexImage2D(GL_TEXTURE_2D, 
            0, 
            GL_RED, 
            self.player.width, 
            self.player.height, 
            0, 
            GL_RED, 
            GL_UNSIGNED_BYTE, 
            None
            )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D,0)

        self.utex=glGenTextures(1)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D,self.utex)
        glTexImage2D(GL_TEXTURE_2D, 
            0, 
            GL_RED, 
            self.player.width//self.uv_scale, 
            self.player.height//self.uv_scale, 
            0, 
            GL_RED, 
            GL_UNSIGNED_BYTE, 
            None
            )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D,0)

        self.vtex=glGenTextures(1)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D,self.vtex)
        glTexImage2D(GL_TEXTURE_2D, 
            0, 
            GL_RED, 
            self.player.width//self.uv_scale, 
            self.player.height//self.uv_scale, 
            0, 
            GL_RED, 
            GL_UNSIGNED_BYTE, 
            None
            )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D,0)

        ## create a framebuffer and attach player texture as color attachment
        self.fbo=glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER,self.fbo)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,self.player_texture)
        glFramebufferTexture2D(GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            self.player_texture,
            0)
        glBindTexture(GL_TEXTURE_2D,0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        ## create PBO
        self.pbosize=(3*self.player.width*self.player.height)//2
        self.pbo1=GLuint(0);glGenBuffers(1,self.pbo1);self.pbo1=self.pbo1.value ## hack to make it work with pyglet
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER,self.pbo1)
        glBufferData(GL_PIXEL_UNPACK_BUFFER,self.pbosize,c_void_p(0),GL_STREAM_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        if self.dual:
            self.pbo2=GLuint(0);glGenBuffers(1,self.pbo2);self.pbo2=self.pbo2.value ## hack to make it work with pyglet
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER,self.pbo2)
            glBufferData(GL_PIXEL_UNPACK_BUFFER,self.pbosize,c_void_p(0),GL_STREAM_DRAW)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        ## even if we are rendering without a vao, we must bind an empty vao
        self.vao=glGenVertexArrays(1)

    def blit(self,unit=GL_TEXTURE0):
        infos=[__gl_unpack__(p) for p in self.player.planes]
        self.pbosize=np.sum([nfo.size_in_bytes for nfo in infos])
        pbo_=next(self.pbos)
        ##orphaning buffer is often cheaper!
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER,pbo_)
        glBufferData(GL_PIXEL_UNPACK_BUFFER,self.pbosize,c_void_p(0),GL_STREAM_DRAW)

        if all([nfo.device_type==1 for nfo in infos]):  ## data is on cpu
            dst_offset=0
            for e,info in enumerate(infos):
                glBufferSubData(GL_PIXEL_UNPACK_BUFFER,
                            dst_offset,  
                            info.width_in_bytes*info.height,
                            cast(info.ptr,POINTER(c_byte))
                            )
                dst_offset+=info.size_in_bytes
                #Image.fromarray(np.ctypeslib.as_array(cast(info.ptr,POINTER(c_byte)),shape=(1440,1440))).show()
        elif all([nfo.device_type==2 for nfo in infos]): ## data is on cuda device
            cuda_pbo = RegisteredBuffer(pbo_)           ## in order to get gpu<->gpu copies in both encoder and decoder, one need to re-register the buffer!
            buffer_mapping = cuda_pbo.map()
            buffptr,buffsize=buffer_mapping.device_ptr_and_size()
            dst_offset=0 ## offset in gl buffer
            for info in infos:
                cpy = pycuda.driver.Memcpy2D()                      ## don't need to recreate it each frame?
                cpy.set_src_device(info.ptr)
                cpy.set_dst_device(buffptr+dst_offset)                       
                cpy.width_in_bytes = info.width_in_bytes ##pbuff.width*pbuff.components
                cpy.src_pitch = info.stride    ## plane.pitch
                cpy.dst_pitch = info.stride    ## because glbuffer is mapped to cuda, it uses the same stride; we later specify glPixelStorei(GL_UNPACK_ROW_LENGTH, self.player.decoded.planes[0].line_size)
                cpy.height = info.height       ## plane.height
                dst_offset+=info.size_in_bytes ##(pbuff.pitch*pbuff.components)*pbuff.height
                cpy(aligned=False)
            pycuda.driver.Context.synchronize()
            buffer_mapping.unmap()

        #update ytex
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,self.ytex)
        glPixelStorei(GL_UNPACK_ROW_LENGTH, infos[0].stride)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0,0, 
            self.player.width, 
            self.player.height,
            GL_RED, 
            GL_UNSIGNED_BYTE, 
            POINTER(c_byte)()
        )
        #glPixelStorei(GL_UNPACK_ROW_LENGTH, 0) ## only required after last transfer!
        #update utex
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D,self.utex)
        glPixelStorei(GL_UNPACK_ROW_LENGTH, infos[1].stride)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0,0, 
            self.player.width//self.uv_scale, 
            self.player.height//self.uv_scale,
            GL_RED, 
            GL_UNSIGNED_BYTE, 
            cast(infos[0].size_in_bytes,POINTER(c_byte))
        )
        #glPixelStorei(GL_UNPACK_ROW_LENGTH, 0) ## only required at the end!
        #update vtex
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D,self.vtex)
        glPixelStorei(GL_UNPACK_ROW_LENGTH, infos[2].stride)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0,0, 
            self.player.width//self.uv_scale, 
            self.player.height//self.uv_scale,
            GL_RED, 
            GL_UNSIGNED_BYTE, 
            cast(infos[0].size_in_bytes+infos[1].size_in_bytes,POINTER(c_byte))
        )
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        ## draw to our framebuffer, ie player texture
        glBindFramebuffer(GL_FRAMEBUFFER,self.fbo)
        glUseProgram(self.program)
        #glUniform1i(glGetUniformLocation(self.program, b'ytex'),0)
        #glUniform1i(glGetUniformLocation(self.program, b'utex'),1)
        #glUniform1i(glGetUniformLocation(self.program, b'vtex'),2)
        glUniform1i(self.ytex_loc,0)
        glUniform1i(self.utex_loc,1)
        glUniform1i(self.vtex_loc,2)
        glUniform1f(self.contrast_loc,self.contrast)
        glUniform1f(self.brightness_loc,self.brightness)
        glBindVertexArray(self.vao)
        glViewport(0, 0, self.player.width, self.player.height)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glBindFramebuffer(GL_FRAMEBUFFER,0)

class GLBridgeRGB:
    def __init__(self,player,player_texture,dual=False):
        self.player=player
        self.player_texture=player_texture
        self.pbosize=self.player.width*self.player.height*3
        self.dual=dual
        self.mkshaders()
        self.mkbuffers()
        if self.dual:
            self.pbos=itertools.cycle([self.pbo1,self.pbo2])
        else:
            self.pbos=itertools.cycle([self.pbo1])
        self.brightness=0.0 ## fake, unused brightness for API
        self.contrast=1.0   ## fake, unused contrast for API

    def __del__(self):
        try:
            self.release()
        except:
            pass

    def release(self):
        ## must be explicilety called. we cannot rely on __del__ as grabage collection being asynchronous, 
        ## the __del__ function may be called after the context has been destroyed
        glDeleteBuffers(1,[self.pbo1])                   # release pbo
        if self.dual:
            glDeleteBuffers(1,[self.pbo2]) 
    
    def mkbuffers(self):
        self.pbosize=3*self.player.width*self.player.height
        self.pbo1=GLuint(0);glGenBuffers(1,self.pbo1);self.pbo1=self.pbo1.value ## hack to make it work with pyglet
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER,self.pbo1)
        glBufferData(GL_PIXEL_UNPACK_BUFFER,self.pbosize,c_void_p(0),GL_STREAM_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        if self.dual:
            self.pbo2=GLuint(0);glGenBuffers(1,self.pbo2);self.pbo2=self.pbo2.value ## hack to make it work with pyglet
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER,self.pbo2)
            glBufferData(GL_PIXEL_UNPACK_BUFFER,self.pbosize,c_void_p(0),GL_STREAM_DRAW)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    def mkshaders(self):
        pass

    def blit(self,unit=GL_TEXTURE0):
        info=__gl_unpack__(self.player.planes[0])
        self.pbosize=info.width_in_bytes*info.height
        pbo_=next(self.pbos)
        ##orphaning buffer is often cheaper!
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER,pbo_)
        glBufferData(GL_PIXEL_UNPACK_BUFFER,self.pbosize,c_void_p(0),GL_STREAM_DRAW)

        if info.device_type==1:  ## data is on cpu
            glBufferSubData(GL_PIXEL_UNPACK_BUFFER,
                            0,  
                            info.width_in_bytes*info.height,
                            cast(info.ptr,POINTER(c_byte))
                            )
        elif info.device_type==2: ## data is on cuda device
            cuda_pbo = RegisteredBuffer(pbo_)           ## in order to get gpu<->gpu copies in both encoder and decoder, one need to re-register the buffer!
            buffer_mapping = cuda_pbo.map()
            buffptr,buffsize=buffer_mapping.device_ptr_and_size()
            cpy = pycuda.driver.Memcpy2D()                      ## don't need to recreate it each frame?
            cpy.set_src_device(info.ptr)
            cpy.set_dst_device(buffptr)                       
            cpy.width_in_bytes = info.width ## make a contiguous buffer on GL side
            cpy.src_pitch = info.stride     ## plane.pitch
            cpy.dst_pitch = info.width      ## make a contiguous buffer on GL side
            cpy.height = info.height        ## plane.height
            cpy(aligned=False)
            pycuda.driver.Context.synchronize()
            buffer_mapping.unmap()

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.player_texture)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, int(pbo_))
        #glPixelStorei(GL_UNPACK_ROW_LENGTH, infos[2].stride)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                            self.player.width, 
                            self.player.height,
                            GL_RGB, GL_UNSIGNED_BYTE, c_void_p(0))
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        '''
        src_plane=self.player._gpu_data().pbuffers[0]
        cuda_pbo = RegisteredBuffer(int(self.pbo)) ## in order to get gpugpu copies in both encoder and decoder, one need to re-register the buffer!
        buffer_mapping = cuda_pbo.map()
        buffptr,buffsize=buffer_mapping.device_ptr_and_size()
        cpy = pycuda.driver.Memcpy2D() ## don't need to recreate it each frame!
        cpy.set_src_device(src_plane.ptr)
        cpy.set_dst_device(buffptr)
        cpy.width_in_bytes = src_plane.width
        cpy.src_pitch = src_plane.pitch
        cpy.dst_pitch = self.player.width*src_plane.components
        cpy.height = src_plane.height
        cpy(aligned=False)
        pycuda.driver.Context.synchronize()
        buffer_mapping.unmap()
        '''
        '''
        dlpack2GL(self.player.cvtSurface.Planes[0], self.pbo)
        ## opengl update texture from pbo
        glActiveTexture(unit)
        glBindTexture(GL_TEXTURE_2D, self.player_texture)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, int(self.pbo))
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                        self.player.width, 
                        self.player.height,
                        GL_RGB, GL_UNSIGNED_BYTE, c_void_p(0))
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        '''