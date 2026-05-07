import cuda.bindings.runtime as cudart
from cuda.core import GraphicsResource, Device
import numpy as np

## tofto write the same for pycuda_utils
class GPUArray:
    '''thin wrapper mimmicking pycuda's GPUArray
    '''
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

def cuda_mem_cpy_2d_from_pbo(pbo,dstptr):
    pass
def cuda_mem_cpy_2d_to_pbo(pbo,srcptr):
    pass