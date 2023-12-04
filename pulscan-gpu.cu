#include <cub/cub.cuh>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

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

// read only cache array of power thresholds, one for each zmax
__constant__ float powerThresholds[16384];


void __global__ splitComplexNumbers(float2 *complexArray, float *realArray, float *imagArray, long arrayLength){
    long globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalThreadIndex < arrayLength) {
        // Split the complex number into its real and imaginary parts
        float real = complexArray[globalThreadIndex].x;
        float complex = complexArray[globalThreadIndex].y;

        // Store the real and imaginary parts in the device arrays
        realArray[globalThreadIndex] = real;
        imagArray[globalThreadIndex] = complex;
    }
}


void __global__ subtractValueFromArrayAndTakeMagnitude(float *array, float value, long arrayLength){
    long globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalThreadIndex < arrayLength) {
        array[globalThreadIndex] = fabsf(array[globalThreadIndex] - value);
    }
}

void __global__ normaliseByMedianAndMAD(float2 *complexArray, float realMedian, float imagMedian, float realMAD, float imagMad, long arrayLength){
    long globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    realMAD = 1.4826 * realMAD;
    imagMad = 1.4826 * imagMad;

    if (globalThreadIndex < arrayLength) {
        complexArray[globalThreadIndex].x = (complexArray[globalThreadIndex].x - realMedian) / realMAD;
        complexArray[globalThreadIndex].y = (complexArray[globalThreadIndex].y - imagMedian) / imagMad;
    }
}

// Utility function to check CUDA calls
inline void checkCuda(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR: %s at %s:%d\n", cudaGetErrorString(result), file, line);
        exit(result);
    }
}
#define CHECK_CUDA(call) checkCuda((call), __FILE__, __LINE__)

// Function to find median of each subarray
float2* readAndNormalizeFFTFileGPU(const char *filepath, long *data_size) {
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
    float* data = (float*) malloc(sizeof(float) * num_floats);
    if (data == NULL) {
        printf("Memory allocation failed\n");
        fclose(f);
        return NULL;
    }

    size_t n = fread(data, sizeof(float), num_floats, f);
    if (n != num_floats) {
        printf("Error reading file\n");
        fclose(f);
        free(data);
        return NULL;
    }
    fclose(f);

    long numComplexFloats = n / 2; 
    float2* complexDataDevice;
    CHECK_CUDA(cudaMalloc((void **) &complexDataDevice, numComplexFloats * sizeof(float2)));
    CHECK_CUDA(cudaMemcpy(complexDataDevice, data, numComplexFloats * sizeof(float2), cudaMemcpyHostToDevice));

    float* realDataDevice;
    float* imagDataDevice;
    CHECK_CUDA(cudaMalloc((void **) &realDataDevice, numComplexFloats * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **) &imagDataDevice, numComplexFloats * sizeof(float)));

    int numThreadsPerBlock = 256;
    long numBlocks = (numComplexFloats + numThreadsPerBlock - 1) / numThreadsPerBlock;
    splitComplexNumbers<<<numBlocks, numThreadsPerBlock>>>(complexDataDevice, realDataDevice, imagDataDevice, numComplexFloats);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Code to sort each subarray and find medians
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    const size_t subArraySize = 131072; // Length of each subarray
    
    // Prepare for sorting
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, realDataDevice, realDataDevice, subArraySize);
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    size_t numSubArrays = (numComplexFloats + subArraySize - 1) / subArraySize;  // Calculate the number of subarrays, including a possibly smaller final subarray
    for (size_t i = 0; i < numSubArrays; i++) {
        size_t currentSubArraySize = subArraySize;

        // Adjust the size for the final subarray if necessary
        if (i == numSubArrays - 1 && numComplexFloats % subArraySize != 0) {
            currentSubArraySize = numComplexFloats % subArraySize;
        }

        // Sort real part of subarray
        float* subArrayStartReal = realDataDevice + i * subArraySize;
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, subArrayStartReal, subArrayStartReal, currentSubArraySize);

        // Sort imaginary part of subarray
        float* subArrayStartImag = imagDataDevice + i * subArraySize;
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, subArrayStartImag, subArrayStartImag, currentSubArraySize);

        // Calculate median for real part
        float medianReal;
        size_t medianIndex = currentSubArraySize / 2;
        CHECK_CUDA(cudaMemcpy(&medianReal, subArrayStartReal + medianIndex, sizeof(float), cudaMemcpyDeviceToHost));
        //printf("Median of real part in subarray %zu: %f\n", i, medianReal);

        // Calculate median for imaginary part
        float medianImag;
        CHECK_CUDA(cudaMemcpy(&medianImag, subArrayStartImag + medianIndex, sizeof(float), cudaMemcpyDeviceToHost));
        //printf("Median of imaginary part in subarray %zu: %f\n", i, medianImag);

        int subArrayThreadsPerBlock = 256;
        int subArrayNumBlocks = (currentSubArraySize + subArrayThreadsPerBlock - 1) / subArrayThreadsPerBlock;
        // Subtract median from each element in subarray
        subtractValueFromArrayAndTakeMagnitude<<<subArrayNumBlocks, subArrayThreadsPerBlock>>>(subArrayStartReal, medianReal, currentSubArraySize);
        subtractValueFromArrayAndTakeMagnitude<<<subArrayNumBlocks, subArrayThreadsPerBlock>>>(subArrayStartImag, medianImag, currentSubArraySize);

        // sort the subarrays
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, subArrayStartReal, subArrayStartReal, currentSubArraySize);
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, subArrayStartImag, subArrayStartImag, currentSubArraySize);

        // Calculate median absolute devation for real part
        float medianAbsoluteDeviationReal;
        CHECK_CUDA(cudaMemcpy(&medianAbsoluteDeviationReal, subArrayStartReal + medianIndex, sizeof(float), cudaMemcpyDeviceToHost));
        //printf("Median absolute deviation of real part in subarray %zu: %f\n", i, medianAbsoluteDeviationReal);

        // Calculate median absolute devation for imaginary part
        float medianAbsoluteDeviationImag;
        CHECK_CUDA(cudaMemcpy(&medianAbsoluteDeviationImag, subArrayStartImag + medianIndex, sizeof(float), cudaMemcpyDeviceToHost));
        //printf("Median absolute deviation of imaginary part in subarray %zu: %f\n", i, medianAbsoluteDeviationImag);

        // Normalise each element in corresponding subarray of complexDataDevice by median and MAD
        normaliseByMedianAndMAD<<<subArrayNumBlocks, subArrayThreadsPerBlock>>>(complexDataDevice + i * subArraySize, medianReal, medianImag, medianAbsoluteDeviationReal, medianAbsoluteDeviationImag, currentSubArraySize);

    }

    CHECK_CUDA(cudaFree(d_temp_storage));

    // Cleanup
    free(data);
    CHECK_CUDA(cudaFree(realDataDevice));
    CHECK_CUDA(cudaFree(imagDataDevice));
    *data_size = numComplexFloats;
    return complexDataDevice;
}




// CUDA kernel to take complex number pair and calculate the squared magnitude 
// i.e. the number multiplied by its complex conjugate (a + bi)(a - bi) = a^2 + b^2

__global__ void complexSquaredMagnitude(float2 *complexArray, float* realArray, long arrayLength) {
    long globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalThreadIndex < arrayLength) {
        // Calculate the squared magnitude of the complex number pair
        float real = complexArray[globalThreadIndex].x;
        float complex = complexArray[globalThreadIndex].y;
        float squaredMagnitude = real * real + complex * complex;
        // Store the squared magnitude in the device array
        realArray[globalThreadIndex] = squaredMagnitude;
    }
}

__global__ void recursiveBoxcar(float *deviceArray, float2 *deviceMax, long n, int zStepSize, int zMax){
    extern __shared__ float sharedMemory[];
    
    // search array should have block size elements, each float2 will be used as [float, int] rather than [float,float]
    float2* searchArray = reinterpret_cast<float2*>(sharedMemory);

    // max array should have block size elements
    float2* maxArray = reinterpret_cast<float2*>(&sharedMemory[2*blockDim.x]);

    // sum array should have block size elements
    float* sumArray = &sharedMemory[4*blockDim.x];

    // lookup array should have block size + zMax elements
    float *lookupArray = &sharedMemory[5*blockDim.x];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long readIndex = threadIdx.x;

    // populate the lookup array
    for (int offset = 0; offset < zMax; offset += blockDim.x) {
        int lookupIndex = readIndex + offset;
        if (lookupIndex < blockDim.x + zMax){
            if (blockIdx.x * blockDim.x + lookupIndex < n) {
                lookupArray[lookupIndex] = deviceArray[blockIdx.x * blockDim.x + lookupIndex];
            } else {
                lookupArray[lookupIndex] = 0;
            }
        }
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
            // so we can reuse the sumArray for the next iteration
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


    if (threadIdx.x*zStepSize < zMax){
        float powerThreshold = powerThresholds[threadIdx.x*zStepSize];
        if (maxArray[threadIdx.x].x > powerThreshold){
            deviceMax[gridDim.x*threadIdx.x + blockIdx.x] = maxArray[threadIdx.x];
        }
    }

    __syncthreads();

}


int boxcarAccelerationSearchExactRBin(float2* deviceComplexArray, int zMax, long inputDataSize, int zStepSize, int numCandidates) {
    int numThreadsPerBlock = 256;
    long numBlocks = (inputDataSize + (long) numThreadsPerBlock - 1) / (long) numThreadsPerBlock;

    numCandidates = numBlocks;

    // Initialise memory for device array of complex numbers as a 1D array of floats: [real, complex, real, complex, ...]

    printf("Starting memory allocation timer\n");
    cudaEvent_t start_mem, stop_mem;
    cudaEventCreate(&start_mem);
    cudaEventRecord(start_mem, 0);

    float2 *deviceMax;
    printf("Allocating deviceMax memory\n");
    cudaMalloc((void **) &deviceMax, sizeof(float2) * (zMax/zStepSize) * numBlocks);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err)); 
    }
    cudaDeviceSynchronize();

    //cuda memset device max to zero
    printf("Zeroing deviceMax memory\n");
    cudaMemset(deviceMax, 0, sizeof(float2) * (size_t) (zMax/zStepSize) * (size_t) numBlocks);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err)); 
    }

    cudaEventCreate(&stop_mem);
    cudaEventRecord(stop_mem, 0);
    cudaEventSynchronize(stop_mem);

    // Calculate the time taken for the memory allocation
    float elapsedTime_mem;
    cudaEventElapsedTime(&elapsedTime_mem, start_mem, stop_mem);
    printf("mem took, %f, ms\n", elapsedTime_mem);

    // begin the timer for the GPU section
    printf("Starting timer\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err)); 
    }

    float* deviceRealArray;
    cudaMalloc((void **) &deviceRealArray, inputDataSize * sizeof(float));

    // Call the kernel to calculate the magnitude of the complex numbers
    printf("Starting Magnitude kernel with numBlocks = %ld and numThreadsPerBlock = %d\n", numBlocks, numThreadsPerBlock);
    complexSquaredMagnitude<<<numBlocks, numThreadsPerBlock>>>(deviceComplexArray, deviceRealArray, inputDataSize);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err)); 
    }

    printf("Starting Boxcar kernel with %ld bytes of shared memory\n", (6*numThreadsPerBlock + zMax) * sizeof(float));
    recursiveBoxcar<<<numBlocks, numThreadsPerBlock, (6*numThreadsPerBlock + zMax) * sizeof(float)>>>(deviceRealArray, deviceMax, inputDataSize, zStepSize, zMax);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err)); 
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
            if (final_output_candidates[i * numCandidates + j].x > 0.0f) {
                fprintf(fp, "%f, %d, %ld\n", final_output_candidates[i * numCandidates + j].x, *reinterpret_cast<int*>(&final_output_candidates[i * numCandidates + j].y), i*zStepSize);
            }
        }
    }

    printf("Finished writing to file\n");

    fclose(fp);

    printf("Closed file\n");

    free(hostMax);

    printf("Freed host memory\n");

    cudaFree(deviceComplexArray);
    cudaFree(deviceMax);

    printf("Freed device memory\n");

    return 0;
}

int main(int argc, char *argv[]) {
    printf("%s", frame);
    if (argc < 2) {
        printf("USAGE: %s file [-zmax int] [-zstep int]\n", argv[0]);
        printf("Required arguments:\n");
        printf("\tfile [string]\tThe input file path (.fft file output of PRESTO realfft)\n");
        printf("Optional arguments:\n");
        printf("\t-zmax [int]\tThe max boxcar width (default = 200)\n");
        printf("\t-zstep [1 or 2]\tThe step size for the boxcar width (default = 2)\n");
        printf("\t-candidates [int]\tThe number of candidates to return per boxcar (default = 10)\n");
        return 1;
    }

    // Get the max_boxcar_width from the command line arguments
    // If not provided, default to 200
    int max_boxcar_width = 200;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-zmax") == 0 && i+1 < argc) {
            max_boxcar_width = atoi(argv[i+1]);
        }
    }

    // Get the zstep from the command line arguments
    // If not provided, default to 2
    int zStepSize = 2;
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
    float2* complexData = readAndNormalizeFFTFileGPU(argv[1], &inputDataNumFloats);

    if(complexData == NULL) {
        printf("Failed to compute magnitudes.\n");
        return 1;
    }

    //Initialise CUDA runtime
    cudaFree(0);

        // copy max_boxcar_width power thresholds from power_thresholds.csv to constant memory
    // the text file will be in the format of a single number of power per line with no header line
    // the line number corresponds to the corresponding z value, so line 0 is z = 0, line 1 is z = 1, etc.
    // cuda constant cache size is 64KB, so we can fit up to 16384 floats in constant memory,
    // so the theoretical cap on max_boxcar_width is 16384

    FILE *fp = fopen("power_thresholds.csv", "r");
    if (fp == NULL) {
        printf("Failed to open power_thresholds.csv for reading.\n");
        return 1;
    }

    printf("Opened power_thresholds.csv\n");

    float powerThresholdsHost[max_boxcar_width];
    for (int i = 0; i < max_boxcar_width; i++) {
        fscanf(fp, "%f", &powerThresholdsHost[i]);
    }

    printf("Finished reading power_thresholds.csv\n");

    fclose(fp);

    // copy powerThresholdsHost to constant memory using cudaMemcpyToSymbol
    cudaMemcpyToSymbol(powerThresholds, powerThresholdsHost, max_boxcar_width * sizeof(float));


    long inputDataNumComplexFloats = inputDataNumFloats / 2;

    boxcarAccelerationSearchExactRBin(complexData, max_boxcar_width, inputDataNumComplexFloats, zStepSize, numCandidates);

    //get last cuda error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err)); 
    }


    return 0;
}