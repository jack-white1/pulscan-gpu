#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

float* read_presto_fft_file(const char *filepath, int *num_floats) {
    //printf("Reading file: %s\n", filepath);

    // Open the file for reading
    FILE *f = fopen(filepath, "rb");
    if (f == NULL) {
        perror("Error opening file");
        return NULL;
    }

    // Determine the file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    // Calculate the number of floats in the file
    *num_floats = file_size / sizeof(float);
    if (*num_floats % 2 != 0) {
        printf("Data file does not contain an even number of floats\n");
        fclose(f);
        return NULL;
    }

    // Allocate memory dynamically based on file size
    float* data = (float*) malloc(file_size);
    if (data == NULL) {
        printf("Memory allocation failed\n");
        fclose(f);
        return NULL;
    }

    // Read the file content into the dynamically allocated memory
    size_t n = fread(data, sizeof(float), *num_floats, f);
    if (n != *num_floats) {
        printf("Error reading data from file\n");
        fclose(f);
        free(data);
        return NULL;
    }

    // compute mean and variance of real and imaginary components, ignoring DC component
    float real_sum = 0.0, imag_sum = 0.0;
    for(int i = 1; i < (int)n / 2; i++) {
        real_sum += data[2 * i];
        imag_sum += data[2 * i + 1];
    }

    float real_mean = real_sum / ((n-1) / 2);
    float imag_mean = imag_sum / ((n-1) / 2);
    float real_variance = 0.0, imag_variance = 0.0;

    for(int i = 1; i < (int)n / 2; i++) {
        real_variance += (data[2 * i] - real_mean)*(data[2 * i] - real_mean);
        imag_variance += (data[2 * i + 1] - imag_mean)*(data[2 * i + 1] - imag_mean);
    }

    real_variance /= ((n-1) / 2);
    imag_variance /= ((n-1) / 2);

    float real_stdev = sqrt(real_variance);
    float imag_stdev = sqrt(imag_variance);

    //printf("Real mean: %f, Real stdev: %f\n", real_mean, real_stdev);
    //printf("Imag mean: %f, Imag stdev: %f\n", imag_mean, imag_stdev);

    for (int i = 0; i < (int) n / 2; i++) {
        data[2*i] = (data[2 * i] - real_mean) / real_stdev;
        data[2*i + 1] = (data[2 * i + 1] - imag_mean) / imag_stdev;
    }

    // zero out DC component
    data[0] = 0.0; // DC component
    data[1] = 0.0; // DC component

    // Close the file
    fclose(f);

    // Return the pointer to the dynamically allocated array
    return data;
}


// CUDA kernel to take complex number pair and calculate the squared magnitude 
// i.e. the number multiplied by its complex conjugate (a + bi)(a - bi) = a^2 + b^2

__global__ void complexSquaredMagnitude(float *deviceComplexArray, float *deviceRealArray, int arrayLength) {
    int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the squared magnitude of the complex number pair
    float real = deviceComplexArray[globalThreadIndex * 2];
    float complex = deviceComplexArray[globalThreadIndex * 2 + 1];
    float squaredMagnitude = real * real + complex * complex;

    if (globalThreadIndex < arrayLength) {
        // Store the squared magnitude in the device array
        deviceRealArray[globalThreadIndex] = squaredMagnitude;
        //printf("Index: %d, Real: %f\n",globalThreadIndex, deviceRealArray[globalThreadIndex]);
    }
}

// CUDA kernel to add a shifted version of deviceLookupArray onto deviceRealArray
__global__ void offsetSum(float *deviceRealArray, float *deviceLookupArray, int offset, int arrayLength) {
    int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset == 0) {
        return;
    }
    if (globalThreadIndex + offset < arrayLength) {
        // Store the squared magnitude in the device array
        deviceRealArray[globalThreadIndex] = deviceRealArray[globalThreadIndex] + deviceLookupArray[globalThreadIndex + offset];
        //printf("Index: %d, Real: %f, Lookup: %f\n",globalThreadIndex, deviceRealArray[globalThreadIndex], deviceLookupArray[globalThreadIndex + offset]);
    }
}

// Device function to perform atomic max, as atomicMax is not defined for floats
__device__ float atomicMax(float* address, float val)
{
    int* addressAsInt = (int*) address;
    int old = *addressAsInt, assumed;
    do {
        assumed = old;
        old = atomicCAS(addressAsInt, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// CUDA kernel to find the maximum value in deviceArray with a given length, and track its index in maxIndex
__global__ void findMaxAndItsIndex(float *deviceArray, float *deviceMax, int *maxIndex, int n, int globalOffset) {
    extern __shared__ float sharedData[];
    int *sharedIndex = (int*)&sharedData[blockDim.x]; 

    int localThreadIndex = threadIdx.x;
    int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalThreadIndex < n) {
        sharedData[localThreadIndex] = deviceArray[globalThreadIndex];
        sharedIndex[localThreadIndex] = globalThreadIndex + globalOffset;
    } else {
        sharedData[localThreadIndex] = -FLT_MAX;
        sharedIndex[localThreadIndex] = -1;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (localThreadIndex < s) {
            if(sharedData[localThreadIndex] < sharedData[localThreadIndex + s]) {
                sharedData[localThreadIndex] = sharedData[localThreadIndex + s];
                sharedIndex[localThreadIndex] = sharedIndex[localThreadIndex + s];
            }
        }
        __syncthreads();
    }

    if (localThreadIndex == 0) {
        atomicMax(deviceMax, sharedData[0]);
        if (*deviceMax == sharedData[0]) {
            *maxIndex = sharedIndex[0];
        }
    }
}

int boxcarAccelerationSearch(float* hostComplexArray, int zMax, int inputDataSize, int numberOfCandidatesPerBoxcar) {
    int totalNumberOfWindows = numberOfCandidatesPerBoxcar;
    int windowWidth = (inputDataSize + totalNumberOfWindows - 1) / totalNumberOfWindows;

    // Initialise memory for device array of complex numbers as a 1D array of floats: [real, complex, real, complex, ...]
    float *deviceComplexArray;
    cudaMalloc((void **) &deviceComplexArray, inputDataSize * 2 * sizeof(float));

    float *deviceRealArray;
    cudaMalloc((void **) &deviceRealArray, inputDataSize * sizeof(float));

    float *deviceLookupArray;
    cudaMalloc((void **) &deviceLookupArray, inputDataSize * sizeof(float));

    float *deviceMax;
    cudaMalloc((void **) &deviceMax, sizeof(float) * zMax * totalNumberOfWindows);

    int *deviceMaxIndex;
    cudaMalloc((void **) &deviceMaxIndex, sizeof(int) * zMax * totalNumberOfWindows);

    // Copy host array to device array
    cudaMemcpy(deviceComplexArray, hostComplexArray, inputDataSize * 2 *sizeof(float), cudaMemcpyHostToDevice);



    // begin the timer for the GPU section
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

    // Call the kernel to calculate the magnitude of the complex numbers
    int numThreadsPerBlock = 256;
    int numBlocks = (inputDataSize + numThreadsPerBlock - 1) / numThreadsPerBlock;
    complexSquaredMagnitude<<<numBlocks, numThreadsPerBlock>>>(deviceComplexArray, deviceRealArray, inputDataSize);
    cudaDeviceSynchronize();

    // Copy deviceRealArray to deviceLookupArray
    cudaMemcpy(deviceLookupArray, deviceRealArray, inputDataSize * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    // Periodicity Search
    int findMaxNumThreadsPerBlock = 512; // MUST BE A POWER OF 2 FOR REDUCTION TO WORK, TUNED TO 512 FOR 4090
    int findMaxNumBlocks = (windowWidth + findMaxNumThreadsPerBlock - 1) / findMaxNumThreadsPerBlock;
    int maxOffset;

    for (int windowIndex = 0; windowIndex < totalNumberOfWindows; windowIndex++) {
        maxOffset = windowIndex;
        findMaxAndItsIndex<<<findMaxNumBlocks, findMaxNumThreadsPerBlock, 2 * findMaxNumThreadsPerBlock * sizeof(float)>>>(&deviceRealArray[windowIndex * windowWidth], deviceMax + maxOffset, deviceMaxIndex + maxOffset, windowWidth, windowIndex * windowWidth);
    }
    cudaDeviceSynchronize();

    // Acceleration Search
    for (int offset = 1; offset < zMax; offset++) {

        // Calculate boxcar filtered spectrum
        offsetSum<<<numBlocks, numThreadsPerBlock>>>(deviceRealArray, deviceLookupArray, offset, inputDataSize);
        cudaDeviceSynchronize();

        // Search individual windows for local maxima
        for (int windowIndex = 0; windowIndex < totalNumberOfWindows; windowIndex++) {
            maxOffset = (totalNumberOfWindows * offset) + windowIndex;
            findMaxAndItsIndex<<<findMaxNumBlocks, findMaxNumThreadsPerBlock, 2 * findMaxNumThreadsPerBlock * sizeof(float)>>>(&deviceRealArray[windowIndex * windowWidth], deviceMax + maxOffset, deviceMaxIndex + maxOffset, windowWidth, windowIndex * windowWidth);
        }
        cudaDeviceSynchronize();
    }

    // End the timer for the cuda kernel
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate the time taken for the kernel to run
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time taken for GPU section to run: %f ms\n", elapsedTime);

    // Copy the maximum values from the device array to the host array
    float *hostMax = (float *) malloc(sizeof(float) * zMax * totalNumberOfWindows);
    cudaMemcpy(hostMax, deviceMax, sizeof(float) * zMax * totalNumberOfWindows, cudaMemcpyDeviceToHost);

    // Copy the maximum index values from the device array to the host array
    int *hostMaxIndex = (int *) malloc(sizeof(int) * zMax * totalNumberOfWindows);
    cudaMemcpy(hostMaxIndex, deviceMaxIndex, sizeof(int) * zMax * totalNumberOfWindows, cudaMemcpyDeviceToHost);

    FILE *fp = fopen("output.csv", "w");
    if (fp == NULL) {
        printf("Failed to open output.csv for writing.\n");
        return 1;
    }

    fprintf(fp, "power, rBin, zBin, zMax*inputDataSize\n"); // Header

    // Print the maximum values to a CSV file
    for (long i = 0; i < zMax; i++) {
        for (long j = 0; j < totalNumberOfWindows; j++) {
            fprintf(fp, "%f, %d, %ld, %ld\n", hostMax[i * totalNumberOfWindows + j], hostMaxIndex[i * totalNumberOfWindows + j], i, (long)zMax * (long)inputDataSize);
        }
    }

    fclose(fp);

    free(hostComplexArray);
    free(hostMax);
    free(hostMaxIndex);

    cudaFree(deviceComplexArray);
    cudaFree(deviceRealArray);
    cudaFree(deviceLookupArray);
    cudaFree(deviceMax);
    cudaFree(deviceMaxIndex);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("USAGE: %s file [-ncpus int] [-zmax int] [-candidates int]\n", argv[0]);
        printf("Required arguments:\n");
        printf("\tfile [string]\tThe input file path (.fft file output of PRESTO realfft)\n");
        printf("Optional arguments:\n");
        printf("\t-zmax [int]\tThe max boxcar width (default = 1200, max = the size of your input data)\n");
        printf("\t-candidates [int]\tThe number of candidates per boxcar (default = 10), total candidates in output will be = zmax * candidates\n");
        return 1;
    }

    // Get the number of candidates per boxcar from the command line arguments
    // If not provided, default to 10
    int candidates_per_boxcar = 10;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-candidates") == 0 && i+1 < argc) {
            candidates_per_boxcar = atoi(argv[i+1]);
        }
    }

    // Get the max_boxcar_width from the command line arguments
    // If not provided, default to 1200
    int max_boxcar_width = 1200;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-zmax") == 0 && i+1 < argc) {
            max_boxcar_width = atoi(argv[i+1]);
        }
    }

    int inputDataNumFloats;
    float* complexData = read_presto_fft_file(argv[1], &inputDataNumFloats);

    if(complexData == NULL) {
        printf("Failed to compute magnitudes.\n");
        return 1;
    }

    int inputDataNumComplexFloats = inputDataNumFloats / 2;

    boxcarAccelerationSearch(complexData, max_boxcar_width, inputDataNumComplexFloats, candidates_per_boxcar);

    return 0;
}
