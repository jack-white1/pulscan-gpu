#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// function to compare floats for qsort
int pulscan_compare_floats_median(const void *a, const void *b) {
    float arg1 = *(const float*)a;
    float arg2 = *(const float*)b;

    if(arg1 < arg2) return -1;
    if(arg1 > arg2) return 1;
    return 0;
}

int pulscan_compare_float2s(const void *a, const void *b) {
    float arg1 = ((float2*)a)->x;
    float arg2 = ((float2*)b)->x;

    if(arg1 < arg2) return 1;
    if(arg1 > arg2) return -1;
    return 0;
}

void pulscan_normalize_block(float* block, size_t block_size) {
    if (block_size == 0) return;

    // Compute the median
    float* sorted_block = (float*) malloc(sizeof(float) * block_size);
    memcpy(sorted_block, block, sizeof(float) * block_size);
    qsort(sorted_block, block_size, sizeof(float), pulscan_compare_floats_median);

    float median;
    if (block_size % 2 == 0) {
        median = (sorted_block[block_size/2 - 1] + sorted_block[block_size/2]) / 2.0f;
    } else {
        median = sorted_block[block_size/2];
    }

    // Compute the MAD
    for (size_t i = 0; i < block_size; i++) {
        sorted_block[i] = fabs(sorted_block[i] - median);
    }
    qsort(sorted_block, block_size, sizeof(float), pulscan_compare_floats_median);

    float mad = block_size % 2 == 0 ?
                (sorted_block[block_size/2 - 1] + sorted_block[block_size/2]) / 2.0f :
                sorted_block[block_size/2];

    free(sorted_block);

    // scale the mad by the constant scale factor k
    float k = 1.4826f; // 1.4826 is the scale factor to convert mad to std dev for a normal distribution https://en.wikipedia.org/wiki/Median_absolute_deviation
    mad *= k;

    // Normalize the block
    if (mad != 0) {
        for (size_t i = 0; i < block_size; i++) {
            block[i] = (block[i] - median) / mad;
        }
    }
}

float* pulscan_readAndNormalizeFFTFile(const char *filepath, long *data_size) {
    size_t block_size = 131072; // needs to be much larger than max boxcar width

    printf("Reading file: %s\n", filepath);

    FILE *f = fopen(filepath, "rb");
    if (f == NULL) {
        perror("Error opening file");
        return NULL;
    }

    // Determine the size of the file
    fseek(f, 0, SEEK_END);
    long filesize = ftell(f);
    fseek(f, 0, SEEK_SET);

    size_t num_floats = filesize / sizeof(float);

    // Allocate memory for the data
    float* data = (float*) malloc(sizeof(float) * num_floats);
    if(data == NULL) {
        printf("Memory allocation failed\n");
        fclose(f);
        return NULL;
    }
    
    size_t n = fread(data, sizeof(float), num_floats, f);
    if (n % 2 != 0) {
        printf("Data file does not contain an even number of floats\n");
        fclose(f);
        free(data);
        return NULL;
    }

    size_t size = n / 2;
    float* magnitude = (float*) malloc(sizeof(float) * size);
    if(magnitude == NULL) {
        printf("Memory allocation failed\n");
        free(data);
        return NULL;
    }

    printf("Finished reading file, normalizing with block size %ld\n", block_size);

    #pragma omp parallel for
    // Perform block normalization
    for (size_t block_start = 0; block_start < size; block_start += block_size) {
        size_t block_end = block_start + block_size < size ? block_start + block_size : size;
        size_t current_block_size = block_end - block_start;

        // Separate the real and imaginary parts
        float* real_block = (float*) malloc(sizeof(float) * current_block_size);
        float* imag_block = (float*) malloc(sizeof(float) * current_block_size);

        if (real_block == NULL || imag_block == NULL) {
            printf("Memory allocation failed for real_block or imag_block\n");
            free(real_block);
            free(imag_block);
        }

        for (size_t i = 0; i < current_block_size; i++) {
            real_block[i] = data[2 * (block_start + i)];
            imag_block[i] = data[2 * (block_start + i) + 1];
        }

        // Normalize real and imaginary parts independently
        pulscan_normalize_block(real_block, current_block_size);
        pulscan_normalize_block(imag_block, current_block_size);

        for (size_t i = 0; i < current_block_size; i++) {
            data[2 * (block_start + i)] = real_block[i];
            data[2 * (block_start + i) + 1] = imag_block[i];
        }

        free(real_block);
        free(imag_block);
    }

    data[0] = 0.0f; // set DC component of spectrum to 0
    data[1] = 0.0f; // set DC component of spectrum to 0

    fclose(f);

    *data_size = (long) size;
    return data;
}


// CUDA kernel to take complex number pair and calculate the squared magnitude 
// i.e. the number multiplied by its complex conjugate (a + bi)(a - bi) = a^2 + b^2

__global__ void pulscan_complexSquaredMagnitudeInPlace(float *deviceArray, long arrayLength) {
    long globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the squared magnitude of the complex number pair
    float real = deviceArray[globalThreadIndex * 2];
    float complex = deviceArray[globalThreadIndex * 2 + 1];
    float squaredMagnitude = real * real + complex * complex;

    __syncthreads();

    if (globalThreadIndex < arrayLength) {
        // Store the squared magnitude in the device array
        deviceArray[globalThreadIndex] = squaredMagnitude;
    }
}

__global__ void pulscan_recursiveBoxcar(float *deviceArray, float2 *deviceMax, long n, int zStepSize, int zMax){
    extern __shared__ float sharedMemory[];
    
    // search array should have block size elements, each float2 will be used as [float, int] rather than [float,float]
    float2* searchArray = reinterpret_cast<float2*>(sharedMemory);

    // max array should have block size elements
    float2* maxArray = reinterpret_cast<float2*>(&sharedMemory[2*blockDim.x]);

    // sum array should have block size elements
    float* sumArray = &sharedMemory[4*blockDim.x];

    // lookup array should have block size + zMax elements
    float *lookupArray = &sharedMemory[5*blockDim.x];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;;

    long readIndex = threadIdx.x;

    while (readIndex < blockDim.x + zMax){
        if (blockIdx.x * blockDim.x + readIndex < n) {
            lookupArray[readIndex] = deviceArray[blockIdx.x * blockDim.x + readIndex];
        } else {
            lookupArray[readIndex] = 0;
        }
        readIndex += blockDim.x;
    }

    __syncthreads();

    // populate the sum array
    sumArray[threadIdx.x] = 0.0f;

    // populate the search array
    searchArray[threadIdx.x].x = sumArray[threadIdx.x];
    searchArray[threadIdx.x].y = *reinterpret_cast<float*>(&idx);


    int outputIndex = 0;
    // begin boxcar filtering
    for (int i = 0; i < zMax; i++){
        __syncthreads();
        // do the offset sum (boxcar filtering)
        sumArray[threadIdx.x] = sumArray[threadIdx.x] + lookupArray[threadIdx.x + i];

        if(i % zStepSize == 0){
            // going to reorder the searchArray, so need to copy the sumArray to the searchArray
            searchArray[threadIdx.x].x = sumArray[threadIdx.x];
            searchArray[threadIdx.x].y = *reinterpret_cast<float*>(&idx);
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) {
                    if(searchArray[threadIdx.x].x < searchArray[threadIdx.x + s].x) {
                        searchArray[threadIdx.x] = searchArray[threadIdx.x + s];
                    }
                }
                __syncthreads();
            }
            
            if (threadIdx.x == 0) {
                maxArray[outputIndex] = searchArray[0];
            }
            outputIndex++;
        }
    }

    __syncthreads();

    if (threadIdx.x < zMax/zStepSize){
        deviceMax[gridDim.x*threadIdx.x + blockIdx.x] = maxArray[threadIdx.x];
    }
}


int pulscan_boxcarAccelerationSearchExactRBin(float* hostComplexArray, int zMax, long inputDataSize, int zStepSize, int numCandidates) {
    int numThreadsPerBlock = 256;
    long numBlocks = (inputDataSize + (long) numThreadsPerBlock - 1) / (long) numThreadsPerBlock;

    // Initialise memory for device array of complex numbers as a 1D array of floats: [real, complex, real, complex, ...]

    printf("Allocating deviceComplexArray memory\n");
    float *deviceComplexArray;
    cudaMalloc((void **) &deviceComplexArray, inputDataSize * 2 * sizeof(float));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err)); 
    }

    // Copy host array to device array
    printf("Copying memory\n");
    cudaMemcpy(deviceComplexArray, hostComplexArray, inputDataSize * 2 *sizeof(float), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err)); 
    }

    printf("Starting memory allocation timer\n");
    cudaEvent_t start_mem, stop_mem;
    cudaEventCreate(&start_mem);
    cudaEventRecord(start_mem, 0);

    float2 *deviceMax;
    printf("Allocating deviceMax memory\n");
    cudaMalloc((void **) &deviceMax, sizeof(float2) * (zMax/zStepSize) * numBlocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err)); 
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err)); 
    }
    cudaDeviceSynchronize();
    //cuda memset device max to zero
    printf("Zeroing deviceMax memory\n");
    cudaMemset(deviceMax, 0, sizeof(float2) * (size_t) (zMax/zStepSize) * (size_t) numBlocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err)); 
    }
    cudaDeviceSynchronize();

    cudaEventCreate(&stop_mem);
    cudaEventRecord(stop_mem, 0);
    cudaEventSynchronize(stop_mem);

    // Calculate the time taken for the memory allocation
    float elapsedTime_mem;
    cudaEventElapsedTime(&elapsedTime_mem, start_mem, stop_mem);
    printf("mem took, %f, ms\n", elapsedTime_mem);

    //for (int run = 0; run < 100; run++) {
    // begin the timer for the GPU section
    printf("Starting timer\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err)); 
    }

    // Call the kernel to calculate the magnitude of the complex numbers
    printf("Starting Magnitude kernel\n");
    pulscan_complexSquaredMagnitudeInPlace<<<numBlocks, numThreadsPerBlock>>>(deviceComplexArray, inputDataSize);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err)); 
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err)); 
    }
    printf("Starting Boxcar kernel\n");
    pulscan_recursiveBoxcar<<<numBlocks, numThreadsPerBlock, (5*numThreadsPerBlock + 2*zMax) * sizeof(float)>>>(deviceComplexArray, deviceMax, inputDataSize, zStepSize, zMax);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err)); 
    }

    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err)); 
    }

    // End the timer for the cuda kernel
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate the time taken for the kernel to run
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Took, %f, ms\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //}
    printf("Finished kernel\n");
    // Copy the maximum values from the device array to the host array
    float2 *hostMax = (float2 *) malloc(sizeof(float2) * (long) (zMax/zStepSize) * numBlocks);
    cudaMemcpy(hostMax, deviceMax, sizeof(float2) * (long) (zMax/zStepSize) * numBlocks, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("Finished copying to host\n");

    float2* final_output_candidates = (float2*) malloc(sizeof(float2) * (long) (zMax/zStepSize) * numCandidates);

    // memset the final output candidates to zero
    memset(final_output_candidates, 0, sizeof(float2) * (long) (zMax/zStepSize) * numCandidates);

    for (int i = 0; i < zMax/zStepSize; i++) {
        for (int j = 0; j < numCandidates; j++) {
            float2 localMax = hostMax[i * numBlocks + j*(numBlocks/numCandidates)];   
            for (int localIndex = 0; localIndex < numBlocks/numCandidates; localIndex++){
                if (hostMax[i * numBlocks + j*(numBlocks/numCandidates) + localIndex].x > localMax.x){
                    localMax = hostMax[i * numBlocks + j*(numBlocks/numCandidates) + localIndex];
                }
            }
            final_output_candidates[i * numCandidates + j] = localMax;
        }
    }


    FILE *fp = fopen("output.csv", "w");
    if (fp == NULL) {
        printf("Failed to open output.csv for writing.\n");
        return 1;
    }

    printf("Opened file\n");

    fprintf(fp, "power, rBin, zBin\n"); // Header



    for (long i = 0; i < zMax/zStepSize; i++) {
        for (long j = 0; j < numCandidates; j++) {
            fprintf(fp, "%f, %d, %ld\n", final_output_candidates[i * numCandidates + j].x, *reinterpret_cast<int*>(&final_output_candidates[i * numCandidates + j].y), i*zStepSize);
        }
    }

    printf("Finished writing to file\n");

    fclose(fp);



    free(hostComplexArray);
    free(hostMax);

    cudaFree(deviceComplexArray);
    cudaFree(deviceMax);

    return 0;
}

#define RESET   "\033[0m"
#define FLASHING   "\033[5m"
#define BOLD   "\033[1m"

const char* pulscan_frame = 
"      .     *    +    .      .   .                 .              *   +  \n"
"  *     " BOLD "____        __" RESET "  +   .             .       " BOLD "__________  __  __" RESET "    .\n"
".   +  " BOLD "/ __ \\__  __/ /_____________" RESET "*" BOLD "_____" RESET "     .  " BOLD "/ ____/ __ \\/ / / /" RESET "     \n"
"      " BOLD "/ /_/ / / / / / ___/ ___/ __ `/ __ \\______/ / __/ /_/ / / / /" RESET "  +   .\n"
"   . " BOLD "/ ____/ /_/ / (__  ) /__/ /_/ / / / /_____/ /_/ / ____/ /_/ /" RESET "\n"
"    " BOLD "/_/" RESET "  . " BOLD "\\__,_/_/____/\\___/\\__,_/_/ /_/" RESET "   *  " BOLD "\\____/_/    \\____/ .  " FLASHING "*" RESET "\n"
" .    .       .   +           .        .         +        .       .      .\n"
"  +     .        .      +       .           .            .    +    .\n"
                                                                

"        J. White, K. Adámek, J. Roy, S. Ransom, W. Armour   2023\n\n";

int main(int argc, char *argv[]) {
    printf("%s", pulscan_frame);
    if (argc < 2) {
        printf("USAGE: %s file [-zmax int] [-zstep int]\n", argv[0]);
        printf("Required arguments:\n");
        printf("\tfile [string]\tThe input file path (.fft file output of PRESTO realfft)\n");
        printf("Optional arguments:\n");
        printf("\t-zmax [int]\tThe max boxcar width (default = 1200)\n");
        printf("\t-zstep [1 or 2]\tThe step size for the boxcar width (default = 2)\n");
        printf("\t-candidates [int]\tThe number of candidates to return per boxcar (default = 10)\n");
        return 1;
    }

    // Get the max_boxcar_width from the command line arguments
    // If not provided, default to 1200
    int max_boxcar_width = 1200;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-zmax") == 0 && i+1 < argc) {
            max_boxcar_width = atoi(argv[i+1]);
        }
    }

    // Get the zstep from the command line arguments
    // If not provided, default to 2
    int zStepSize = 1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-zstep") == 0 && i+1 < argc) {
            zStepSize = atoi(argv[i+1]);
        }
    }

    // Get the number of candidates to return per boxcar from the command line arguments
    // If not provided, default to 10
    int numCandidates = 10;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-candidates") == 0 && i+1 < argc) {
            numCandidates = atoi(argv[i+1]);
        }
    }

    long inputDataNumFloats;
    float* complexData = pulscan_readAndNormalizeFFTFile(argv[1], &inputDataNumFloats);

    if(complexData == NULL) {
        printf("Failed to compute magnitudes.\n");
        return 1;
    }

    long inputDataNumComplexFloats = inputDataNumFloats / 2;

    pulscan_boxcarAccelerationSearchExactRBin(complexData, max_boxcar_width, inputDataNumComplexFloats, zStepSize, numCandidates);

    return 0;
}