@del /Q release\CavesOfTitan.exe
@CL /Os /Iinclude main.cpp /link lib\glfw3dll.lib opengl32.lib lib\opencl.lib /out:release/CavesOfTitan.exe
@del main.obj
@del /Q release\kernels
@del /Q release\shaders
@del /Q release\images
@del /Q release\sfx
@xcopy kernels release\kernels /i /E
@xcopy shaders release\shaders /i /E
@xcopy images release\images /i /E
@xcopy sfx release\sfx /i /E