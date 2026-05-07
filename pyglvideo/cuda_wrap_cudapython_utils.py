import cuda.bindings.runtime as cudart
from cuda.core import GraphicsResource, Device
import numpy as np

##
## thin wrappers around cuda-python and pycuda api
##
class GPUArray:
    def __init__(self,size,dtype=np.uint8):
        w=size[1]
        h=size[0]
        err,ptr,pitch=cudart.cudaMallocPitch(w*np.dtype(dtype).itemsize,h)
        if not err:
            self.ptr=ptr
            self.width=w
            self.height=h
            self.strides=(pitch,1)
            self.dtype=np.dtype(np.uint8).str

    def __del__(self):
        cudart.cuMemFree(self.cuda_y)

    @property
    def __cuda_array_interface__(self):
        return {'shape':(self.height,self.width),
                'typestr':self.dtype,
                'data':(self.ptr,False),
                'strides':self.strides,
                'stream':None,
                'version':3
                }
                
class CUDAMappedPBOPtr:
    def __init__(self,pbo):
        self.pbo=pbo
        _dev=Device(0)
        _dev.set_current()
        self.stream=_dev.create_stream()
        self.resource=GraphicsResource.from_gl_buffer(pbo, flags="write_discard")
    def __enter__(self):
        buffer=self.resource.map(stream=self.stream)
        return buffer.handle
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.resource.unmap()

def cuda_memcpy_2d(src,dst, width, height,spitch,dpitch,**kwargs):
    cudart.cudaMemcpy2D(src=src,
                        dst=dst,
                        width=width,
                        spitch=spitch,
                        dpitch=dpitch,
                        height=height,
                        kind=cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
                        )
