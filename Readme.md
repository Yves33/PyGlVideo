## PyGLVideo

### About
A utility python library to display video files in OpenGL textures and record openGL textures into video files. The library can optionnaly perform GPUdecode->openGL->GPUencode without copying data to the CPU (which is its main interest).

### Requirements
The library supports two backends (PyAV and Vali), at least one must be installed.
+ PyopenGL
+ [PyAV](https://github.com/pyav-org/pyav) (at least version 17 is required dor __dlpack__ support)

For hardware decode/encode and direct GPU<->GL tansfer, you will need the following
+ [Vali](https://github.com/RomanArzumanyan/VALI) (optionnal, depends on the backend)
+ [pycuda](https://github.com/inducer/pycuda) compiled with openGL support (the distributed versions on conda and pip do not have GL support)
+ [dlpack](https://github.com/pearu/pydlpack)

For the demo program:  
+ [moderngl](https://github.com/moderngl/moderngl) and [moderngl-window](https://github.com/moderngl/moderngl-window)
+ imgui (either [pyimgui](https://github.com/pyimgui/pyimgui) or [imgui-bundle](https://github.com/pthom/imgui_bundle))

### Installation & usage
Just drop pyglvideo folder in your repo. running the test program requires to set the environment variable "TEST_VIDEO_FILE" to the path of a valid video/image.

### Notes:
+ The file moderngl_window_integrations_imgui_bundle.py is a pending pull release for compatibility between moderngl-window and imgui-bundle. 
+ Using direct gpu<->gl transfers becomes interesting for hi res videos (4k and more). Don't expect major speed improvements with low resolution videos
+ Consider the library as beta software
+ When using full hardware pipelines, vali and av backends achieve comparable speed. In my experience, VALI is slower when decoding multiple streams at the same time.

### Todo:
+ Provide CudaBridgeRGB for transfering RGB(a) textures to cuda.
