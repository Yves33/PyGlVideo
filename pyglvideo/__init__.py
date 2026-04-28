import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*g_value_get_string.*")
import logging
logging.getLogger().setLevel(logging.ERROR)

from .glbridge import GLBridgeRGB,GLBridgeYUV420,GLBridgeNV12,GLBridgeOneshot,CudaBridgeNV12

try:
    from .avgrabber import FrameGrabberAV
    from .avrecorder import FrameRecorderAV
    AV_AVAILABLE=True
except:
    AV_AVAILABLE=False

try:
    from .valigrabber import FrameGrabberVALI
    from .valirecorder import FrameRecorderVALI
    VALI_AVAILABLE=True
except:
    VALI_AVAILABLE=False

try:
    from .pilgrabber import FrameGrabberPIL
    PIL_AVAILABLE=True
except:
    PIL_AVAILABLE=False

def FrameGrabber(*args, **kwargs):
    av_kwargs=['hwaccel','is_hw_owned']
    pil_kwargs=['fps','framecount']
    vali_kwargs=['gpuid','is_hw_owned','tgt_format']
    nv_kwargs=['gpuid','is_hw_owned','tgt_format']

    backend=kwargs.pop('backend','auto')
    if backend=='av' and AV_AVAILABLE:
        logging.info("Using AV (ffmpeg) backend")
        localkwargs={k:kwargs[k] for k in av_kwargs if k in kwargs.keys()}
        return FrameGrabberAV(*args, **localkwargs)
    elif backend=='pil' and PIL_AVAILABLE:
        logging.info("Using PILLOW (image) backend")
        localkwargs={k:kwargs[k] for k in pil_kwargs if k in kwargs.keys()}
        return FrameGrabberPIL(*args, **localkwargs)
    elif backend=='vali' and PIL_AVAILABLE:
        logging.info("Using NVIDIA VALI backend")
        localkwargs={k:kwargs[k] for k in vali_kwargs if k in kwargs.keys()}
        return FrameGrabberVALI(*args, **localkwargs)
    ## NV backend is currently disabled and not implemented as it is not clear if it provides any performance advantage over VALI
    #if backend=='nv' and NV_AVAILABLE:
    #    from .nvgrabber import FrameGrabberNV
    #    logging.info("Using NV (pyNvvideocodec) backend")
    #    localkwargs={k:kwargs[k] for k in av_kwargs if k in kwargs.keys()}
    #    return FrameGrabberNV(*args, **localkwargs)

def FrameRecorder(*args, **kwargs):
    av_kwargs=['width','height','codec','fps','options','hwaccel']
    pil_kwargs=['fps','framecount']
    vali_kwargs=['gpuid','width','height','codec','fps','options','device']

    backend=kwargs.pop('backend','auto')
    if backend=='av' and AV_AVAILABLE:
        logging.info("Using AV (ffmpeg) backend")
        localkwargs={k:v for k,v in kwargs.items() if k in av_kwargs}
        return FrameRecorderAV(*args, **localkwargs)
    elif backend=='vali' and VALI_AVAILABLE:
        logging.info("Using NVIDIA VALI backend")
        localkwargs={k:v for k,v in kwargs.items() if k in vali_kwargs}
        return FrameRecorderVALI(*args, **localkwargs)
