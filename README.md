1. Download using `git clone https://github.com/jack-white1/pulscan-gpu`

compile with `nvcc pulscan-gpu.cu -o pulscan-gpu -lstdc++ -lm`

run with `./pulscan-gpu <input file>` e.g. `./pulscan-gpu test.fft`