#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>


// load the .fft file into CPU ram. it is a binary file of 32 bit floats, interleaved real and imaginary parts

// copy it to the GPU, trimmed down to the nearest factor of N elements

// separate the interleaved real and imaginary parts

// block normalise them, maybe using thrust?
    // find the median of each subarray of 32768 elements in the real and imaginary parts
    // subtract the median from each element in the subarray, take the absolute value of this value
    // find the median of the absolute values, multiply it by 1.486
    // in the original real and imaginary parts, subtract the median and divide through by the median absolute deviation of of the relevant subarray

// take the magnitude of the complex numbers 

// decimate x2, x3, x4 into 3 separate arrays

// boxcar filter each one, each threadblock returns 1 value for every other zstep
    // this is therefore a data reduction by a factor of 2 * blockWidth.x

struct candidate{
    float power;
    float logp;
    int blockIndex;
    int z;
};

double __device__ power_to_logp(float chi2, float dof){
    double double_dof = (double) dof;
    double double_chi2 = (double) chi2;
    // Use boundary condition
    if (dof >= chi2 * 1.05){
        return 0.0;
    } else {
        double x = 1500 * double_dof / double_chi2;
        // Updated polynomial equation
        double f_x = (-4.460405902717228e-46 * pow(x, 16) + 9.492786384945832e-42 * pow(x, 15) - 
               9.147045144529116e-38 * pow(x, 14) + 5.281085384219971e-34 * pow(x, 13) - 
               2.0376166670276118e-30 * pow(x, 12) + 5.548033164083744e-27 * pow(x, 11) - 
               1.0973877021703706e-23 * pow(x, 10) + 1.5991806841151474e-20 * pow(x, 9) - 
               1.7231488066853853e-17 * pow(x, 8) + 1.3660070957914896e-14 * pow(x, 7) - 
               7.861795249869729e-12 * pow(x, 6) + 3.2136336591718867e-09 * pow(x, 5) - 
               9.046641813341226e-07 * pow(x, 4) + 0.00016945948004599545 * pow(x, 3) - 
               0.0214942314851717 * pow(x, 2) + 2.951595476316614 * x - 
               755.240918031251);
        double logp = chi2 * f_x / 1500;
        return logp;
    }
}

__global__ void separateRealAndImaginaryComponents(float2* rawDataDevice, float* realData, float* imaginaryData, long numComplexFloats){
    long globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;
    if (globalThreadIndex < numComplexFloats){
        float2 currentValue = rawDataDevice[globalThreadIndex];
        realData[globalThreadIndex] = currentValue.x;
        imaginaryData[globalThreadIndex] = currentValue.y;
    }
}

__global__ void medianOfMediansNormalisation(float* globalArray, float* median, float* mad) {
    // Each thread loads 4 elements from global memory to shared memory
    // Assumes blockDim.x = 1024
    // TODO: make this work for any blockDim.x
    __shared__ float medianArray[4096];
    __shared__ float madArray[4096];
    __shared__ float normalisedArray[4096];

    int globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;
    int localThreadIndex = threadIdx.x;

    medianArray[localThreadIndex] = globalArray[globalThreadIndex];
    medianArray[localThreadIndex + 1024] = globalArray[globalThreadIndex + 1024];
    medianArray[localThreadIndex + 2048] = globalArray[globalThreadIndex + 2048];
    medianArray[localThreadIndex + 3072] = globalArray[globalThreadIndex + 3072];

    madArray[localThreadIndex] = medianArray[localThreadIndex];
    madArray[localThreadIndex + 1024] = medianArray[localThreadIndex + 1024];
    madArray[localThreadIndex + 2048] = medianArray[localThreadIndex + 2048];
    madArray[localThreadIndex + 3072] = medianArray[localThreadIndex + 3072];

    normalisedArray[localThreadIndex] = medianArray[localThreadIndex];
    normalisedArray[localThreadIndex + 1024] = medianArray[localThreadIndex + 1024];
    normalisedArray[localThreadIndex + 2048] = medianArray[localThreadIndex + 2048];
    normalisedArray[localThreadIndex + 3072] = medianArray[localThreadIndex + 3072];

    __syncthreads();

    float a,b,c,d,min,max;
  
    a = medianArray[localThreadIndex];
    b = medianArray[localThreadIndex+1024];
    c = medianArray[localThreadIndex+2048];
    d = medianArray[localThreadIndex+3072];
    min = fminf(fminf(fminf(a,b),c),d);
    max = fmaxf(fmaxf(fmaxf(a,b),c),d);
    medianArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    __syncthreads();

    if(localThreadIndex < 256){
        a = medianArray[localThreadIndex];
        b = medianArray[localThreadIndex+256];
        c = medianArray[localThreadIndex+512];
        d = medianArray[localThreadIndex+768];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        medianArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex < 64){
        a = medianArray[localThreadIndex];
        b = medianArray[localThreadIndex+64];
        c = medianArray[localThreadIndex+128];
        d = medianArray[localThreadIndex+192];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        medianArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex < 16){
        a = medianArray[localThreadIndex];
        b = medianArray[localThreadIndex+16];
        c = medianArray[localThreadIndex+32];
        d = medianArray[localThreadIndex+48];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        medianArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex < 4){
        a = medianArray[localThreadIndex];
        b = medianArray[localThreadIndex+4];
        c = medianArray[localThreadIndex+8];
        d = medianArray[localThreadIndex+12];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        medianArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex == 0){
        a = medianArray[localThreadIndex];
        b = medianArray[localThreadIndex+1];
        c = medianArray[localThreadIndex+2];
        d = medianArray[localThreadIndex+3];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        medianArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
        *median = medianArray[0];
    }

    __syncthreads();

    madArray[localThreadIndex] = fabsf(madArray[localThreadIndex] - *median);
    madArray[localThreadIndex + 1024] = fabsf(madArray[localThreadIndex + 1024] - *median);
    madArray[localThreadIndex + 2048] = fabsf(madArray[localThreadIndex + 2048] - *median);
    madArray[localThreadIndex + 3072] = fabsf(madArray[localThreadIndex + 3072] - *median);

    __syncthreads();

    a = madArray[localThreadIndex];
    b = madArray[localThreadIndex+1024];
    c = madArray[localThreadIndex+2048];
    d = madArray[localThreadIndex+3072];
    min = fminf(fminf(fminf(a,b),c),d);
    max = fmaxf(fmaxf(fmaxf(a,b),c),d);
    madArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    __syncthreads();

    if(localThreadIndex < 256){
        a = madArray[localThreadIndex];
        b = madArray[localThreadIndex+256];
        c = madArray[localThreadIndex+512];
        d = madArray[localThreadIndex+768];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        madArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex < 64){
        a = madArray[localThreadIndex];
        b = madArray[localThreadIndex+64];
        c = madArray[localThreadIndex+128];
        d = madArray[localThreadIndex+192];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        madArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex < 16){
        a = madArray[localThreadIndex];
        b = madArray[localThreadIndex+16];
        c = madArray[localThreadIndex+32];
        d = madArray[localThreadIndex+48];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        madArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex < 4){
        a = madArray[localThreadIndex];
        b = madArray[localThreadIndex+4];
        c = madArray[localThreadIndex+8];
        d = madArray[localThreadIndex+12];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        madArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
    }
    __syncthreads();
    if(localThreadIndex == 0){
        a = madArray[localThreadIndex];
        b = madArray[localThreadIndex+1];
        c = madArray[localThreadIndex+2];
        d = madArray[localThreadIndex+3];
        min = fminf(fminf(fminf(a,b),c),d);
        max = fmaxf(fmaxf(fmaxf(a,b),c),d);
        madArray[localThreadIndex] = (a+b+c+d-min-max)*0.5;
        *mad = 1/(madArray[0]*1.4826);
    }
    __syncthreads();

    normalisedArray[localThreadIndex] = (normalisedArray[localThreadIndex] - *median) * *mad;
    normalisedArray[localThreadIndex + 1024] = (normalisedArray[localThreadIndex + 1024] - *median) * *mad;
    normalisedArray[localThreadIndex + 2048] = (normalisedArray[localThreadIndex + 2048] - *median) * *mad;
    normalisedArray[localThreadIndex + 3072] = (normalisedArray[localThreadIndex + 3072] - *median) * *mad;

    __syncthreads();

    globalArray[globalThreadIndex] = normalisedArray[localThreadIndex];
    globalArray[globalThreadIndex + 1024] = normalisedArray[localThreadIndex + 1024];
    globalArray[globalThreadIndex + 2048] = normalisedArray[localThreadIndex + 2048];
    globalArray[globalThreadIndex + 3072] = normalisedArray[localThreadIndex + 3072];

}

__global__ void magnitudeSquared(float* realData, float* imaginaryData, float* magnitudeSquaredArray, long numFloats){
    int globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;
    if (globalThreadIndex < numFloats){
        float real = realData[globalThreadIndex];
        float imaginary = imaginaryData[globalThreadIndex];
        magnitudeSquaredArray[globalThreadIndex] = real*real + imaginary*imaginary;
    }
}


// takes a 1D array like this:
// magnitudeSquaredArray:   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
// and adds these elements together, effectively performing a harmonic sum
// decimatedArray2:         [0,0,0,0,0,x,0,0,0,0,0,x,x,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
// decimatedArray3:         [0,0,0,0,0,x,0,0,0,0,0,x,x,0,0,0,0,x,x,x,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
// decimatedArray4:         [0,0,0,0,0,x,0,0,0,0,0,x,x,0,0,0,0,x,x,x,0,0,0,x,x,x,x,0,0,0,0,0,0,0,0,0]

__global__ void decimateHarmonics(float* magnitudeSquaredArray, float* decimatedArray2, float* decimatedArray3, float* decimatedArray4, long numMagnitudes){
    int globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;

    float a,b,c,d,e,f,g,h,i,j;

    if (globalThreadIndex*2+1 < numMagnitudes){
        a = magnitudeSquaredArray[globalThreadIndex];
        b = magnitudeSquaredArray[globalThreadIndex*2];
        c = magnitudeSquaredArray[globalThreadIndex*2+1];
        decimatedArray2[globalThreadIndex] = a+b+c;
    }

    if (globalThreadIndex*3+2 < numMagnitudes){
        d = magnitudeSquaredArray[globalThreadIndex*3];
        e = magnitudeSquaredArray[globalThreadIndex*3+1];
        f = magnitudeSquaredArray[globalThreadIndex*3+2];
        decimatedArray3[globalThreadIndex] = a+b+c+d+e+f;
    }

    if (globalThreadIndex*4+3 < numMagnitudes){
        g = magnitudeSquaredArray[globalThreadIndex*4];
        h = magnitudeSquaredArray[globalThreadIndex*4+1];
        i = magnitudeSquaredArray[globalThreadIndex*4+2];
        j = magnitudeSquaredArray[globalThreadIndex*4+3];
        decimatedArray4[globalThreadIndex] = a+b+c+d+e+f+g+h+i+j;
    }
}

// logarithmic zstep, zmax = 256, numThreads = 256
__global__ void boxcarFilterArray(float* magnitudeSquaredArray, candidate* globalCandidateArray, long numFloats){
    __shared__ float lookupArray[512];
    __shared__ float sumArray[256];
    __shared__ float searchArray[256];
    __shared__ candidate localCandidateArray[16];

    int globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;
    int localThreadIndex = threadIdx.x;

    lookupArray[localThreadIndex] = magnitudeSquaredArray[globalThreadIndex];
    lookupArray[localThreadIndex + 256] = magnitudeSquaredArray[globalThreadIndex + 256];

    __syncthreads();

    // initialise the sum array
    sumArray[localThreadIndex] = lookupArray[localThreadIndex];
    __syncthreads();
    // begin boxcar filtering
    int targetZ = 1;
    int outputCounter = 0;

    for (int z = 0; z < 256; z+=1){
        sumArray[localThreadIndex] +=  lookupArray[localThreadIndex + z];
        if (z = targetZ){
            searchArray[localThreadIndex] = sumArray[localThreadIndex];
            for (int stride = blockDim.x / 2; stride>0; stride >>= 1){
                if (localThreadIndex < stride){
                    searchArray[localThreadIndex] = fmaxf(searchArray[localThreadIndex], searchArray[localThreadIndex + stride]);
                }
                __syncthreads();
            }
            localCandidateArray[outputCounter].power = searchArray[0];
            localCandidateArray[outputCounter].blockIndex = blockIdx.x;
            localCandidateArray[outputCounter].z = z;
            localCandidateArray[outputCounter].logp = 0.0f;
            outputCounter+=1;
            targetZ *= 2;
        }
        __syncthreads();
    }

    __syncthreads();

    if (localThreadIndex < 16){
        globalCandidateArray[blockIdx.x*16+localThreadIndex] = localCandidateArray[localThreadIndex];
    }
}

__global__ void calculateLogp(candidate* globalCandidateArray, long numCandidates){
    int globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;
    if (globalThreadIndex < numCandidates){
        double logp = power_to_logp(globalCandidateArray[globalThreadIndex].power,globalCandidateArray[globalThreadIndex].z);
        globalCandidateArray[globalThreadIndex].logp = (float) logp;
    }

}

#define RESET   "\033[0m"
#define FLASHING   "\033[5m"
#define BOLD   "\033[1m"

const char* frame = 
"      .     *    +    .      .   .                 .              *   +  \n"
"  *     " BOLD "____        __" RESET "  +   .             .       " BOLD "__________  __  __" RESET "    .\n"
".   +  " BOLD "/ __ \\__  __/ /_____________" RESET "*" BOLD "_____" RESET "     .  " BOLD "/ ____/ __ \\/ / / /" RESET "     \n"
"      " BOLD "/ /_/ / / / / / ___/ ___/ __ `/ __ \\______/ / __/ /_/ / / / /" RESET "  +   .\n"
"   . " BOLD "/ ____/ /_/ / (__  ) /__/ /_/ / / / /_____/ /_/ / ____/ /_/ /" RESET "\n"
"    " BOLD "/_/" RESET "  . " BOLD "\\__,_/_/____/\\___/\\__,_/_/ /_/" RESET "   *  " BOLD "\\____/_/    \\____/ .  " FLASHING "*" RESET "\n"
" .    .       .   +           .        .         +        .       .      .\n"
"  +     .        .      +       .           .            .    +    .\n"
"        J. White, K. Adámek, J. Roy, S. Ransom, W. Armour   2023\n\n";

int main(int argc, char* argv[]){
    printf("%s", frame);

    // start high resolution timer to measure gpu initialisation time using chrono
    auto start_chrono = std::chrono::high_resolution_clock::now();
    
    cudaDeviceSynchronize();

    auto end_chrono = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono - start_chrono);
    
    printf("GPU initialisation took:                %f ms\n",(float)duration.count());
    
    // start timing
    start_chrono = std::chrono::high_resolution_clock::now();

    if (argc < 2) {
        printf("Please provide the input file path as a command line argument.\n");
        return 1;
    }

    const char* filepath = argv[1];

    FILE *f = fopen(filepath, "rb");

    // Determine the size of the file in bytes
    fseek(f, 0, SEEK_END);
    size_t filesize = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    // Read the file into CPU memory
    size_t numFloats = filesize / sizeof(float);

    // Cap the filesize at the nearest lower factor of 8192
    numFloats = numFloats - (numFloats % 8192);
    float* rawData = (float*) malloc(sizeof(float) * numFloats);
    fread(rawData, sizeof(float), numFloats, f);
    fclose(f);
    
    // stop timing
    end_chrono = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono - start_chrono);
    printf("Reading file took:                      %f ms\n", (float)duration.count());

    // start GPU timing
    cudaEvent_t start, stop;
    // start timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Allocate a suitable array on the GPU, copy raw data across
    float* rawDataDevice;
    cudaMalloc((void**)&rawDataDevice, sizeof(float) * numFloats);
    cudaMemcpy(rawDataDevice, rawData, sizeof(float) * numFloats, cudaMemcpyHostToDevice);

    // stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Copying data to GPU took:               %f ms\n", milliseconds);

    // start timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int numMagnitudes = numFloats/2;

    // Separate the interleaved real and imaginary parts as the raw data
    // is in the format [[real,imaginary], [real,imaginary]]
    // this is basically a transpose from a 2xN to an Nx2 array
    float* realDataDevice;
    float* imaginaryDataDevice;
    cudaMalloc((void**)&realDataDevice, sizeof(float)*numMagnitudes);
    cudaMalloc((void**)&imaginaryDataDevice, sizeof(float)*numMagnitudes);

    int numThreadsSeparate = 256;
    int numBlocksSeparate = (numMagnitudes + numThreadsSeparate - 1)/ numThreadsSeparate;
    separateRealAndImaginaryComponents<<<numBlocksSeparate, numThreadsSeparate>>>((float2*)rawDataDevice, realDataDevice, imaginaryDataDevice, numMagnitudes);

    // stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Separating complex components took:     %f ms\n", milliseconds);

    // start timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Normalise the real and imaginary parts
    float* medianReal;
    float* madReal;
    cudaMalloc((void**)&medianReal, sizeof(float));
    cudaMalloc((void**)&madReal, sizeof(float));

    int numMagnitudesPerThreadNormalise = 4;
    int numThreadsNormalise = 1024; // DO NOT CHANGE THIS, TODO: make it changeable
    int numBlocksNormalise = ((numMagnitudes/(numMagnitudesPerThreadNormalise * numThreadsNormalise)) + numThreadsNormalise - 1)/ numThreadsNormalise;

    medianOfMediansNormalisation<<<numBlocksNormalise, numThreadsNormalise>>>(realDataDevice, medianReal, madReal);
    medianOfMediansNormalisation<<<numBlocksNormalise, numThreadsNormalise>>>(imaginaryDataDevice, medianReal, madReal);
    cudaDeviceSynchronize();

    
    // stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Normalisation took:                     %f ms\n", milliseconds);

    // start timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Take the magnitude of the complex numbers
    float* magnitudeSquaredArray;
    cudaMalloc((void**)&magnitudeSquaredArray, sizeof(float)*numMagnitudes);

    int numThreadsMagnitude = 1024;
    int numBlocksMagnitude = (numMagnitudes + numThreadsMagnitude - 1)/ numThreadsMagnitude;

    magnitudeSquared<<<numBlocksMagnitude, numThreadsMagnitude>>>(realDataDevice, imaginaryDataDevice, magnitudeSquaredArray, numMagnitudes);
    cudaDeviceSynchronize();
    
    // stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Magnitude took:                         %f ms\n", milliseconds);

    // start timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    float* decimatedArrayBy2;
    float* decimatedArrayBy3;
    float* decimatedArrayBy4;
    cudaMalloc((void**)&decimatedArrayBy2, sizeof(float)*numMagnitudes/2);
    cudaMalloc((void**)&decimatedArrayBy3, sizeof(float)*numMagnitudes/3);
    cudaMalloc((void**)&decimatedArrayBy4, sizeof(float)*numMagnitudes/4);

    int numThreadsDecimate = 256;
    int numBlocksDecimate = (numMagnitudes/2 + numThreadsDecimate - 1)/ numThreadsDecimate;

    decimateHarmonics<<<numBlocksDecimate, numThreadsDecimate>>>(realDataDevice, decimatedArrayBy2, decimatedArrayBy3, decimatedArrayBy4, numMagnitudes);
    cudaDeviceSynchronize();
    
    // stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Decimation took:                        %f ms\n", milliseconds);

    // start timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int numThreadsBoxcar = 256;
    int numBlocksBoxcar1 = (numMagnitudes + numThreadsBoxcar - 1)/ numThreadsBoxcar;
    int numBlocksBoxcar2 = (numMagnitudes/2 + numThreadsBoxcar - 1)/ numThreadsBoxcar;
    int numBlocksBoxcar3 = (numMagnitudes/3 + numThreadsBoxcar - 1)/ numThreadsBoxcar;
    int numBlocksBoxcar4 = (numMagnitudes/4 + numThreadsBoxcar - 1)/ numThreadsBoxcar;

    candidate* globalCandidateArray1;
    candidate* globalCandidateArray2;
    candidate* globalCandidateArray3;
    candidate* globalCandidateArray4;

    cudaMalloc((void**)&globalCandidateArray1, sizeof(candidate)*16*numBlocksBoxcar1);
    cudaMalloc((void**)&globalCandidateArray2, sizeof(candidate)*16*numBlocksBoxcar2);
    cudaMalloc((void**)&globalCandidateArray3, sizeof(candidate)*16*numBlocksBoxcar3);
    cudaMalloc((void**)&globalCandidateArray4, sizeof(candidate)*16*numBlocksBoxcar4);
    
    boxcarFilterArray<<<numBlocksBoxcar1, numThreadsBoxcar>>>(magnitudeSquaredArray, globalCandidateArray1, numMagnitudes);
    boxcarFilterArray<<<numBlocksBoxcar2, numThreadsBoxcar>>>(decimatedArrayBy2, globalCandidateArray2, numMagnitudes/2);
    boxcarFilterArray<<<numBlocksBoxcar3, numThreadsBoxcar>>>(decimatedArrayBy3, globalCandidateArray3, numMagnitudes/3);
    boxcarFilterArray<<<numBlocksBoxcar4, numThreadsBoxcar>>>(decimatedArrayBy4, globalCandidateArray4, numMagnitudes/4);
    cudaDeviceSynchronize();

    
    // stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Boxcar filtering took:                  %f ms\n", milliseconds);

    // start timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int numThreadsLogp = 256;
    int numBlocksLogp1 = (numBlocksBoxcar1*16 + numThreadsLogp - 1)/ numThreadsLogp;
    int numBlocksLogp2 = (numBlocksBoxcar2*16 + numThreadsLogp - 1)/ numThreadsLogp;
    int numBlocksLogp3 = (numBlocksBoxcar3*16 + numThreadsLogp - 1)/ numThreadsLogp;
    int numBlocksLogp4 = (numBlocksBoxcar4*16 + numThreadsLogp - 1)/ numThreadsLogp;

    calculateLogp<<<numBlocksLogp1, numThreadsLogp>>>(globalCandidateArray1, numBlocksBoxcar1*16);
    calculateLogp<<<numBlocksLogp2, numThreadsLogp>>>(globalCandidateArray2, numBlocksBoxcar2*16);
    calculateLogp<<<numBlocksLogp3, numThreadsLogp>>>(globalCandidateArray3, numBlocksBoxcar3*16);
    calculateLogp<<<numBlocksLogp4, numThreadsLogp>>>(globalCandidateArray4, numBlocksBoxcar4*16);
    cudaDeviceSynchronize();

    // stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Logp time taken:                        %f ms\n", milliseconds);


    free(rawData);
    cudaFree(rawDataDevice);
    cudaFree(realDataDevice);
    cudaFree(imaginaryDataDevice);
    cudaFree(medianReal);
    cudaFree(madReal);

    // check last cuda error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    return 0;

}