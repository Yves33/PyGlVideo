import pycuda
from pycuda.gl import RegisteredBuffer
from pycuda.gpuarray import GPUArray ## requires pytools

##
## thin wrappers around cuda-python and pycuda api
##
#class GPUArray(GPUArray):
                
class CUDAMappedPBOPtr:
    def __init__(self,pbo):
        self.cuda_pbo=RegisteredBuffer(pbo)
    def __enter__(self):
        self.buffer_mapping = self.cuda_pbo.map()
        ptr,buffsize=self.buffer_mapping.device_ptr_and_size()
        return ptr
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.buffer_mapping.unmap()

def cuda_memcpy_2d(src,dst, width, height,spitch,dpitch,**kwargs):
    cpy = pycuda.driver.Memcpy2D()
    cpy.set_src_device(src)
    cpy.set_dst_device(dst)                       
    cpy.width_in_bytes = width
    cpy.src_pitch = spitch
    cpy.dst_pitch = dpitch
    cpy.height = height
    cpy(aligned=False)
    pycuda.driver.Context.synchronize()
