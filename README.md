1. Download using `git clone https://github.com/jack-white1/pulscan-gpu`

compile with `nvcc pulscan_gpu.cu -o pulscan_gpu -lstdc++ -lm`
or for GH200 version `nvcc localcdflib.o ipmpar.o pulscan_GH200.cu -o pulscan_GH200 -Xcompiler -fopenmp`

run with `./pulscan_gpu <input file>` e.g. `./pulscan_gpu sample_data/J1227-4853_GMRT_DM43.40.fft` or `./pulscan_GH200 sample_data/J1227-4853_GMRT_DM43.40.fft`