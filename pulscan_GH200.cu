#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>

extern "C" {
#include "localcdflib.h"
}

double log_asymtotic_incomplete_gamma(double a, double z)
/*
  log_asymtotic_incomplete_gamma(double a, double z):
      Return the natural log of the incomplete gamma function in
          its asymtotic limit as z->infty.  This is from Abramowitz
          and Stegun eqn 6.5.32.
*/
{
    double x = 1.0, newxpart = 1.0, term = 1.0;
    int ii = 1;

    //printf("log_asymtotic_incomplete_gamma() being called with arguments:\n");
    //printf("   a = %f, z = %f\n", a, z);

    while (fabs(newxpart) > 1e-15) {
        term *= (a - ii);
        newxpart = term / pow(z, ii);
        x += newxpart;
        ii += 1;
        //printf("ii = %d, x = %f, newxpart = %f\n", ii, x, newxpart);
    }
    //printf("Took %d iterations.\n", ii);
    return (a - 1.0) * log(z) - z + log(x);
}

double log_asymtotic_gamma(double z)
/*
  log_asymtotic_gamma(double z):
      Return the natural log of the gamma function in its asymtotic limit
          as z->infty.  This is from Abramowitz and Stegun eqn 6.1.41.
*/
{
    double x, y;
    //printf("log_asymtotic_gamma() being called with argument z = %f\n", z);
    x = (z - 0.5) * log(z) - z + 0.91893853320467267;
    y = 1.0 / (z * z);
    x += (((-5.9523809523809529e-4 * y
            + 7.9365079365079365079365e-4) * y
           - 2.7777777777777777777778e-3) * y + 8.3333333333333333333333e-2) / z;
    return x;
}
double extended_equiv_gaussian_sigma(double logp)
/*
  extended_equiv_gaussian_sigma(double logp):
      Return the equivalent gaussian sigma corresponding to the 
          natural log of the cumulative gaussian probability logp.
          In other words, return x, such that Q(x) = p, where Q(x)
          is the cumulative normal distribution.  This version uses
          the rational approximation from Abramowitz and Stegun,
          eqn 26.2.23.  Using the log(P) as input gives a much
          extended range.
*/
{
    double t, num, denom;

    t = sqrt(-2.0 * logp);
    num = 2.515517 + t * (0.802853 + t * 0.010328);
    denom = 1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308));
    return t - num / denom;
}


double chi2_logp(double chi2, double dof)
/* MODIFIED FOR PULSCAN TO CLOSE INVALID REGION AT HIGH DOF */
/* Return the natural log probability corresponding to a chi^2 value */
/* of chi2 given dof degrees of freedom. */
{
    double logp;
    //printf("chi2 = %f, dof = %f\n", chi2, dof);

    if (chi2 <= 0.0) {
        return -INFINITY;
    }
    //printf("chi2/dof = %f\n", chi2/dof);
    // COMMENT OUT NEXT LINE IS THE MODIFICATION
    //if (chi2 / dof > 15.0 || (dof > 150 && chi2 / dof > 6.0)) {
    if (chi2 / dof > 1.0) {
        //printf("chi2/dof > 1.0\n");
        // printf("Using asymtotic expansion...\n");
        // Use some asymtotic expansions for the chi^2 distribution
        //   this is eqn 26.4.19 of A & S
        logp = log_asymtotic_incomplete_gamma(0.5 * dof, 0.5 * chi2) -
            log_asymtotic_gamma(0.5 * dof);
    } else {
        //printf("chi2/dof <= 1.0\n");
        int which, status;
        double p, q, bound, df = dof, x = chi2;

        which = 1;
        status = 0;
        // Determine the basic probability
        cdfchi(&which, &p, &q, &x, &df, &status, &bound);
        if (status) {
            printf("\nError in cdfchi() (chi2_logp()):\n");
            printf("   status = %d, bound = %g\n", status, bound);
            printf("   p = %g, q = %g, x = %g, df = %g\n\n", p, q, x, df);
            exit(1);
        }
        // printf("p = %.3g  q = %.3g\n", p, q);
        logp = log(q);
    }
    return logp;
}

double equivalent_gaussian_sigma(double logp)
/* Return the approximate significance in Gaussian sigmas */
/* corresponding to a natural log probability logp        */
{
    double x;

    if (logp < -600.0) {
        x = extended_equiv_gaussian_sigma(logp);
    } else {
        int which, status;
        double p, q, bound, mean = 0.0, sd = 1.0;
        q = exp(logp);
        p = 1.0 - q;
        which = 2;
        status = 0;
        /* Convert to a sigma */
        cdfnor(&which, &p, &q, &x, &mean, &sd, &status, &bound);
        if (status) {
            if (status == -2) {
                x = 0.0;
            } else if (status == -3) {
                x = 38.5;
            } else {
                printf("\nError in cdfnor() (candidate_sigma()):\n");
                printf("   status = %d, bound = %g\n", status, bound);
                printf("   p = %g, q = %g, x = %g, mean = %g, sd = %g\n\n",
                       p, q, x, mean, sd);
                exit(1);
            }
        }
    }
    if (x < 0.0)
        return 0.0;
    else
        return x;
}

double candidate_sigma(double power, int numsum, double numtrials)
/* Return the approximate significance in Gaussian       */
/* sigmas of a candidate of numsum summed powers,        */
/* taking into account the number of independent trials. */
{
    //printf("candidate_sigma() being called with arguments:\n");
    //printf("   power = %f, numsum = %d, numtrials = %f\n",
    //       power, numsum, numtrials);
    double logp, chi2, dof;

    if (power <= 0.0) {
        return 0.0;
    }

    // Get the natural log probability
    chi2 = 2.0 * power;
    dof = 2.0 * numsum;
    logp = chi2_logp(chi2, dof);

    // Correct for numtrials
    logp += log(numtrials);

    // Convert to sigma
    return equivalent_gaussian_sigma(logp);
}

/*
 *  This Quickselect routine is based on the algorithm described in
 *  "Numerical recipies in C", Second Edition,
 *  Cambridge University Press, 1992, Section 8.5, ISBN 0-521-43108-5
*/

/* Fast computation of the median of an array. */
/* Note:  It messes up the order!              */

#define ELEM_SWAP(a,b) { register float t=(a);(a)=(b);(b)=t; }

float median_function(float arr[], int n)
{
    int low, high;
    int median;
    int middle, ll, hh;

    low = 0;
    high = n - 1;
    median = (low + high) / 2;
    for (;;) {
        if (high <= low)        /* One element only */
            return arr[median];

        if (high == low + 1) {  /* Two elements only */
            if (arr[low] > arr[high])
                ELEM_SWAP(arr[low], arr[high]);
            return arr[median];
        }

        /* Find median of low, middle and high items; swap into position low */
        middle = (low + high) / 2;
        if (arr[middle] > arr[high])
            ELEM_SWAP(arr[middle], arr[high]);
        if (arr[low] > arr[high])
            ELEM_SWAP(arr[low], arr[high]);
        if (arr[middle] > arr[low])
            ELEM_SWAP(arr[middle], arr[low]);

        /* Swap low item (now in position middle) into position (low+1) */
        ELEM_SWAP(arr[middle], arr[low + 1]);

        /* Nibble from each end towards middle, swapping items when stuck */
        ll = low + 1;
        hh = high;
        for (;;) {
            do
                ll++;
            while (arr[low] > arr[ll]);
            do
                hh--;
            while (arr[hh] > arr[low]);

            if (hh < ll)
                break;

            ELEM_SWAP(arr[ll], arr[hh]);
        }

        /* Swap middle item (in position low) back into correct position */
        ELEM_SWAP(arr[low], arr[hh]);

        /* Re-set active partition */
        if (hh <= median)
            low = ll;
        if (hh >= median)
            high = hh - 1;
    }
}

#undef ELEM_SWAP

void normalize_block_quickselect(float* block, size_t block_size) {
    if (block_size == 0) return;

    // Allocate memory for a copy of the block
    float* sorted_block = (float*) malloc(sizeof(float) * block_size);

    // Copy the block to sorted_block
    memcpy(sorted_block, block, sizeof(float) * block_size);

    // Compute the median using the new function
    float median = median_function(sorted_block, block_size);
    //printf("Median: %f\n", median);

    // Compute the MAD
    for (size_t i = 0; i < block_size; i++) {
        sorted_block[i] = fabs(sorted_block[i] - median); // Calculate the absolute deviation from the median
    }

    // Re-compute the median of the deviations to get the MAD
    float mad = median_function(sorted_block, block_size);
    //printf("MAD: %f\n", mad);

    // Free the allocated memory
    free(sorted_block);

    // Scale the MAD by the constant scale factor k
    float k = 1.4826f; // Scale factor to convert MAD to standard deviation for a normal distribution
    mad *= k;

    // Normalize the block
    if (mad != 0) {
        for (size_t i = 0; i < block_size; i++) {
            block[i] = (block[i] - median) / mad;
        }
    }
}

// TODO CUDA streams




struct candidate{
    float power;
    float logp;
    int r;
    int z;
    int numharm;
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

// takes a 1D array like this:
// magnitudeSquaredArray:   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
// and adds these elements together, effectively performing a harmonic sum
// decimatedArray2:         [0,0,0,0,0,x,0,0,0,0,0,x,x,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
// decimatedArray3:         [0,0,0,0,0,x,0,0,0,0,0,x,x,0,0,0,0,x,x,x,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
// decimatedArray4:         [0,0,0,0,0,x,0,0,0,0,0,x,x,0,0,0,0,x,x,x,0,0,0,x,x,x,x,0,0,0,0,0,0,0,0,0]
//                                     |<--------->|<--------->|<--------->|
//                                        equal spacing between harmonics

__global__ void decimateHarmonics(float* magnitudeSquaredArray, float* decimatedArray2, float* decimatedArray3, float* decimatedArray4, long numMagnitudes){
    int globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;

    float fundamental;
    float harmonic1a, harmonic1b;
    float harmonic2a, harmonic2b, harmonic2c;
    float harmonic3a, harmonic3b, harmonic3c, harmonic3d;

    if (globalThreadIndex*2+1 < numMagnitudes){
        fundamental = magnitudeSquaredArray[globalThreadIndex];
        harmonic1a = magnitudeSquaredArray[globalThreadIndex*2];
        harmonic1b = magnitudeSquaredArray[globalThreadIndex*2+1];
        decimatedArray2[globalThreadIndex] = fundamental+harmonic1a+harmonic1b;
    }

    if (globalThreadIndex*3+2 < numMagnitudes){
        harmonic2a = magnitudeSquaredArray[globalThreadIndex*3];
        harmonic2b = magnitudeSquaredArray[globalThreadIndex*3+1];
        harmonic2c = magnitudeSquaredArray[globalThreadIndex*3+2];
        decimatedArray3[globalThreadIndex] = fundamental+harmonic1a+harmonic1b
                                                +harmonic2a+harmonic2b+harmonic2c;
    }

    if (globalThreadIndex*4+3 < numMagnitudes){
        harmonic3a = magnitudeSquaredArray[globalThreadIndex*4];
        harmonic3b = magnitudeSquaredArray[globalThreadIndex*4+1];
        harmonic3c = magnitudeSquaredArray[globalThreadIndex*4+2];
        harmonic3d = magnitudeSquaredArray[globalThreadIndex*4+3];
        decimatedArray4[globalThreadIndex] = fundamental+harmonic1a+harmonic1b
                                                +harmonic2a+harmonic2b+harmonic2c
                                                +harmonic3a+harmonic3b+harmonic3c+harmonic3d;
    }

    //if (globalThreadIndex == 50000){
    //    printf("fundamental: %f, harmonic1a: %f, harmonic1b: %f, harmonic2a: %f, harmonic2b: %f, harmonic2c: %f, harmonic3a: %f, harmonic3b: %f, harmonic3c: %f, harmonic3d: %f\n", fundamental, harmonic1a, harmonic1b, harmonic2a, harmonic2b, harmonic2c, harmonic3a, harmonic3b, harmonic3c, harmonic3d);
    //}
}

// logarithmic zstep, zmax = 256, numThreads = 256
// TODO: any zmax
// TODO: any blockWidth
// will need to use dynamically allocated shared memory
__global__ void boxcarFilterArray(float* magnitudeSquaredArray, candidate* globalCandidateArray, int numharm, long numFloats){
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
            localCandidateArray[outputCounter].r = blockIdx.x*blockDim.x;
            localCandidateArray[outputCounter].z = z;
            localCandidateArray[outputCounter].logp = 0.0f;
            localCandidateArray[outputCounter].numharm = numharm;
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

__global__ void calculateLogp(candidate* globalCandidateArray, long numCandidates, int numSum){
    int globalThreadIndex = blockDim.x*blockIdx.x + threadIdx.x;
    if (globalThreadIndex < numCandidates){
        double logp = power_to_logp(globalCandidateArray[globalThreadIndex].power,globalCandidateArray[globalThreadIndex].z*numSum);
        globalCandidateArray[globalThreadIndex].logp = (float) logp;
    }
}

void copyDeviceArrayToHostAndPrint(float* deviceArray, long numFloats){
    float* hostArray;
    hostArray = (float*)malloc(sizeof(float)*numFloats);
    cudaMemcpy(hostArray, deviceArray, sizeof(float)*numFloats,cudaMemcpyDeviceToHost);
    for (int i = 0; i < numFloats; i++){
        printf("%f\n", hostArray[i]);
    }
    free(hostArray);
}

void copyDeviceArrayToHostAndSaveToFile(float* deviceArray, long numFloats, const char* filename){
    float* hostArray;
    hostArray = (float*)malloc(sizeof(float)*numFloats);
    cudaMemcpy(hostArray, deviceArray, sizeof(float)*numFloats,cudaMemcpyDeviceToHost);
    FILE *f = fopen(filename, "wb");
    // write in csv format, one number per column
    for (int i = 0; i < numFloats; i++){
        fprintf(f, "%f\n", hostArray[i]);
    }
    fclose(f);
    free(hostArray);
}

float* compute_magnitude_block_normalization_mad(const char *filepath, int *magnitude_size, int ncpus) {
    // begin timer for reading input file
    double start = omp_get_wtime();
    size_t block_size = 32768; // needs to be much larger than max boxcar width

    //printf("Reading file: %s\n", filepath);

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

    double end = omp_get_wtime();
    double time_spent = end - start;
    printf("Reading the data took      %f seconds using 1 thread\n", time_spent);

    start = omp_get_wtime();

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
        normalize_block_quickselect(real_block, current_block_size);
        normalize_block_quickselect(imag_block, current_block_size);

        // Recompute the magnitudes after normalization
        for (size_t i = block_start; i < block_end; i++) {
            magnitude[i] = real_block[i - block_start] * real_block[i - block_start] +
                        imag_block[i - block_start] * imag_block[i - block_start];
        }

        free(real_block);
        free(imag_block);
    }

    magnitude[0] = 0.0f; // set DC component of magnitude spectrum to 0

    fclose(f);
    free(data);

    *magnitude_size = (int) size;

    end = omp_get_wtime();
    time_spent = end - start;
    printf("Normalizing the data took  %f seconds using %d thread(s)\n", time_spent, ncpus);
    return magnitude;
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
    int debug = 0;
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

    int ncpus = 72;

    // define filepath variable
    const char* filepath = argv[1];

    int magnitude_array_size;
    float* magnitudes = compute_magnitude_block_normalization_mad(filepath, &magnitude_array_size, ncpus);
    if (magnitudes == NULL) {
        printf("Error reading the input file.\n");
        return 1;
    }

    // stop timing
    end_chrono = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono - start_chrono);
    printf("Magnitude took:                         %f ms\n",(float)duration.count());


    int numMagnitudes = magnitude_array_size;

    float* magnitudeSquaredArray;
    cudaMalloc((void**)&magnitudeSquaredArray, sizeof(float)*numMagnitudes);

    // start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    cudaMemcpy(magnitudeSquaredArray, magnitudes, sizeof(float)*numMagnitudes, cudaMemcpyHostToDevice);



    //copyDeviceArrayToHostAndPrint(magnitudeSquaredArray, numMagnitudes);
    //copyDeviceArrayToHostAndSaveToFile(magnitudeSquaredArray, numMagnitudes, "magnitudeSquaredArray.bin");

    // stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
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


    if (debug == 1) {
        printf("Calling decimateHarmonics with %d blocks and %d threads per block\n", numBlocksDecimate, numThreadsDecimate);
    }
    decimateHarmonics<<<numBlocksDecimate, numThreadsDecimate>>>(magnitudeSquaredArray, decimatedArrayBy2, decimatedArrayBy3, decimatedArrayBy4, numMagnitudes);
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

    
    if (debug == 1) {
        printf("Calling boxcarFilterArray with %d blocks and %d threads per block\n", numBlocksBoxcar1, numThreadsBoxcar);
        printf("Calling boxcarFilterArray with %d blocks and %d threads per block\n", numBlocksBoxcar2, numThreadsBoxcar);
        printf("Calling boxcarFilterArray with %d blocks and %d threads per block\n", numBlocksBoxcar3, numThreadsBoxcar);
        printf("Calling boxcarFilterArray with %d blocks and %d threads per block\n", numBlocksBoxcar4, numThreadsBoxcar);
    }
    boxcarFilterArray<<<numBlocksBoxcar1, numThreadsBoxcar>>>(magnitudeSquaredArray, globalCandidateArray1, 1, numMagnitudes);
    boxcarFilterArray<<<numBlocksBoxcar2, numThreadsBoxcar>>>(decimatedArrayBy2, globalCandidateArray2, 2, numMagnitudes/2);
    boxcarFilterArray<<<numBlocksBoxcar3, numThreadsBoxcar>>>(decimatedArrayBy3, globalCandidateArray3, 3, numMagnitudes/3);
    boxcarFilterArray<<<numBlocksBoxcar4, numThreadsBoxcar>>>(decimatedArrayBy4, globalCandidateArray4, 4, numMagnitudes/4);
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

    if (debug == 1) {
        printf("Calling calculateLogp with %d blocks and %d threads per block\n", numBlocksLogp1, numThreadsLogp);
        printf("Calling calculateLogp with %d blocks and %d threads per block\n", numBlocksLogp2, numThreadsLogp);
        printf("Calling calculateLogp with %d blocks and %d threads per block\n", numBlocksLogp3, numThreadsLogp);
        printf("Calling calculateLogp with %d blocks and %d threads per block\n", numBlocksLogp4, numThreadsLogp);
    }
    calculateLogp<<<numBlocksLogp1, numThreadsLogp>>>(globalCandidateArray1, numBlocksBoxcar1*16, 1);
    calculateLogp<<<numBlocksLogp2, numThreadsLogp>>>(globalCandidateArray2, numBlocksBoxcar2*16, 3);
    calculateLogp<<<numBlocksLogp3, numThreadsLogp>>>(globalCandidateArray3, numBlocksBoxcar3*16, 6);
    calculateLogp<<<numBlocksLogp4, numThreadsLogp>>>(globalCandidateArray4, numBlocksBoxcar4*16, 10);
    cudaDeviceSynchronize();

    // stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Logp time taken:                        %f ms\n", milliseconds);

    // start chrono timer for writing output file
    start_chrono = std::chrono::high_resolution_clock::now();

    candidate* hostCandidateArray1;
    candidate* hostCandidateArray2;
    candidate* hostCandidateArray3;
    candidate* hostCandidateArray4;

    hostCandidateArray1 = (candidate*)malloc(sizeof(candidate)*16*numBlocksBoxcar1);
    hostCandidateArray2 = (candidate*)malloc(sizeof(candidate)*16*numBlocksBoxcar2);
    hostCandidateArray3 = (candidate*)malloc(sizeof(candidate)*16*numBlocksBoxcar3);
    hostCandidateArray4 = (candidate*)malloc(sizeof(candidate)*16*numBlocksBoxcar4);

    cudaMemcpy(hostCandidateArray1, globalCandidateArray1, sizeof(candidate)*16*numBlocksBoxcar1, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostCandidateArray2, globalCandidateArray2, sizeof(candidate)*16*numBlocksBoxcar2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostCandidateArray3, globalCandidateArray3, sizeof(candidate)*16*numBlocksBoxcar3, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostCandidateArray4, globalCandidateArray4, sizeof(candidate)*16*numBlocksBoxcar4, cudaMemcpyDeviceToHost);

    // output filename is inputfilename with the .fft stripped and replaced with .gpucand
    char outputFilename[256];
    strncpy(outputFilename, filepath, strlen(filepath) - 4);
    outputFilename[strlen(filepath) - 4] = '\0';
    strcat(outputFilename, ".gpucand");


    // write the candidates to a csv file with a header line
    //FILE *csvFile = fopen("gpucandidates.csv", "w");
    FILE *csvFile = fopen(outputFilename, "w");
    fprintf(csvFile, "r,z,power,logp,numharm\n");

    float logpThreshold = -50;

    for (int i = 0; i < numBlocksBoxcar1*16; i++){
        if (i % 16 < 9){
            if (hostCandidateArray1[i].logp < logpThreshold){
                if (hostCandidateArray1[i].r != 0){
                    fprintf(csvFile, "%d,%d,%f,%f,%d\n", hostCandidateArray1[i].r, hostCandidateArray1[i].z, hostCandidateArray1[i].power, hostCandidateArray1[i].logp, hostCandidateArray1[i].numharm);
                }
            }
        }
    }
    
    for (int i = 0; i < numBlocksBoxcar2*16; i++){
        if (i % 16 < 9){
            if (hostCandidateArray2[i].logp < logpThreshold){
                if (hostCandidateArray1[i].r != 0){
                    fprintf(csvFile, "%d,%d,%f,%f,%d\n", hostCandidateArray2[i].r, hostCandidateArray2[i].z, hostCandidateArray2[i].power, hostCandidateArray2[i].logp, hostCandidateArray2[i].numharm);
                }
            }
        }
    }

    for (int i = 0; i < numBlocksBoxcar3*16; i++){
        if (i % 16 < 9){
            if (hostCandidateArray3[i].logp < logpThreshold){
                if (hostCandidateArray1[i].r != 0){
                    fprintf(csvFile, "%d,%d,%f,%f,%d\n", hostCandidateArray3[i].r, hostCandidateArray3[i].z, hostCandidateArray3[i].power, hostCandidateArray3[i].logp, hostCandidateArray3[i].numharm);
                }
            }
        }
    }

    for (int i = 0; i < numBlocksBoxcar4*16; i++){
        if (i % 16 < 9){
            if (hostCandidateArray4[i].logp < logpThreshold){
                if (hostCandidateArray1[i].r != 0){
                    fprintf(csvFile, "%d,%d,%f,%f,%d\n", hostCandidateArray4[i].r, hostCandidateArray4[i].z, hostCandidateArray4[i].power, hostCandidateArray4[i].logp, hostCandidateArray4[i].numharm);
                }
            }
        }
    }

    fclose(csvFile);

    // stop chrono timer for writing output file
    end_chrono = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_chrono - start_chrono);
    printf("Writing output file took:               %f ms\n", (float)duration.count());

    cudaFree(magnitudeSquaredArray);
    cudaFree(decimatedArrayBy2);
    cudaFree(decimatedArrayBy3);
    cudaFree(decimatedArrayBy4);
    cudaFree(globalCandidateArray1);
    cudaFree(globalCandidateArray2);
    cudaFree(globalCandidateArray3);
    cudaFree(globalCandidateArray4);

    // check last cuda error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    return 0;
}

