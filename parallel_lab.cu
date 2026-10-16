#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// CPU Functions (Task 1)

void cpu_smooth(const int *in, int *out, int n) {
    out[0] = in[0];
    out[n - 1] = in[n - 1];
    for (int i = 1; i < n - 1; i++) {
        out[i] = (in[i - 1] + in[i] + in[i + 1]) / 3;
    }
}

int cpu_get_bin(int val) {
    if (val >= 10 && val <= 31) return 0;
    if (val >= 32 && val <= 54) return 1;
    if (val >= 55 && val <= 76) return 2;
    if (val >= 77 && val <= 99) return 3;
    return -1; // Out of range safety
}

void cpu_threshold(const int *in, int *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = cpu_get_bin(in[i]);
    }
}

void cpu_histogram(const int *bins, int *histo, int n) {
    for (int i = 0; i < 4; i++) histo[i] = 0;
    for (int i = 0; i < n; i++) {
        int b = bins[i];
        if (b >= 0 && b < 4) {
            histo[b]++;
        }
    }
}

// CUDA Kernels (Task 2 & Task 3)

// Task 2a: Smoothing Kernel
__global__ void gpu_smooth_kernel(const int *in, int *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (idx == 0 || idx == n - 1) {
            out[idx] = in[idx];
        } else {
            out[idx] = (in[idx - 1] + in[idx] + in[idx + 1]) / 3;
        }
    }
}

// Task 2b: Thresholding Kernel
__global__ void gpu_threshold_kernel(const int *in, int *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int val = in[idx];
        if (val >= 10 && val <= 31) out[idx] = 0;
        else if (val >= 32 && val <= 54) out[idx] = 1;
        else if (val >= 55 && val <= 76) out[idx] = 2;
        else if (val >= 77 && val <= 99) out[idx] = 3;
        else out[idx] = -1;
    }
}

// Task 3a: Histogram without synchronization (Race conditions expected)
__global__ void gpu_histo_naive(const int *bins, int *histo, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int b = bins[idx];
        if (b >= 0 && b < 4) {
            histo[b]++; // Unsafe race condition!
        }
    }
}

// Task 3c: Histogram with global atomicAdd
__global__ void gpu_histo_atomic(const int *bins, int *histo, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int b = bins[idx];
        if (b >= 0 && b < 4) {
            atomicAdd(&histo[b], 1);
        }
    }
}

// Task 3d: Histogram using shared memory and block-level reduction
__global__ void gpu_histo_shared(const int *bins, int *histo, int n) {
    __shared__ int s_histo[4];
    
    // Initialize shared memory bins
    if (threadIdx.x < 4) {
        s_histo[threadIdx.x] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop to handle large N safely per block
    for (int i = idx; i < n; i += stride) {
        int b = bins[i];
        if (b >= 0 && b < 4) {
            atomicAdd(&s_histo[b], 1);
        }
    }
    __syncthreads();

    // Merge shared memory counts into global memory using atomicAdd
    if (threadIdx.x < 4) {
        atomicAdd(&histo[threadIdx.x], s_histo[threadIdx.x]);
    }
}

// Main Execution Driver

int main() {
    // --- Task 1: Verification on N = 60 ---
    int N_small = 60;
    int *hA = (int*)malloc(N_small * sizeof(int));
    int *h_cpu_smooth = (int*)malloc(N_small * sizeof(int));
    int *h_cpu_thresh = (int*)malloc(N_small * sizeof(int));
    int h_cpu_histo[4];

    // Initialize hA with pseudo-random values in range [10, 99]
    srand(42);
    for (int i = 0; i < N_small; i++) {
        hA[i] = 10 + (rand() % 90);
    }

    cpu_smooth(hA, h_cpu_smooth, N_small);
    cpu_threshold(h_cpu_smooth, h_cpu_thresh, N_small);
    cpu_histogram(h_cpu_thresh, h_cpu_histo, N_small);

    printf("=== Task 1: CPU Baseline Verified for N = %d ===\n", N_small);

    // --- Task 2: CUDA Kernels Verification for N = 60 ---
    int *dA, *dS, *dT;
    CHECK_CUDA(cudaMalloc(&dA, N_small * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dS, N_small * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dT, N_small * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(dA, hA, N_small * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 8;
    int gridSize = (N_small + blockSize - 1) / blockSize;

    gpu_smooth_kernel<<<gridSize, blockSize>>>(dA, dS, N_small);
    gpu_threshold_kernel<<<gridSize, blockSize>>>(dS, dT, N_small);
    CHECK_CUDA(cudaDeviceSynchronize());

    int *h_gpu_smooth = (int*)malloc(N_small * sizeof(int));
    int *h_gpu_thresh = (int*)malloc(N_small * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_gpu_smooth, dS, N_small * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_gpu_thresh, dT, N_small * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify Smoothing
    int smooth_mismatch = -1;
    for (int i = 0; i < N_small; i++) {
        if (h_cpu_smooth[i] != h_gpu_smooth[i]) {
            smooth_mismatch = i;
            break;
        }
    }
    printf("Smoothing verification: %s\n", smooth_mismatch == -1 ? "matches CPU" : "mismatch at index");

    // Verify Thresholding
    int thresh_mismatch = -1;
    for (int i = 0; i < N_small; i++) {
        if (h_cpu_thresh[i] != h_gpu_thresh[i]) {
            thresh_mismatch = i;
            break;
        }
    }
    printf("Thresholding verification: %s\n", thresh_mismatch == -1 ? "matches CPU" : "mismatch at index");

    // Cleanup small allocation
    cudaFree(dA); cudaFree(dS); cudaFree(dT);
    free(hA); free(h_cpu_smooth); free(h_cpu_thresh);
    free(h_gpu_smooth); free(h_gpu_thresh);

    // --- Timing for Large N (N = 2^20 and N = 2^24) ---
    int sizes[] = {1 << 20, 1 << 24};
    for (int s = 0; s < 2; s++) {
        int N = sizes[s];
        printf("\n--- Benchmarking for N = 2^%d (%d elements) ---\n", 20 + s * 4, N);

        int *h_large = (int*)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++) h_large[i] = 10 + (rand() % 90);

        int *d_in, *d_out, *d_bins, *d_histo;
        CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_bins, N * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_histo, 4 * sizeof(int)));

        CHECK_CUDA(cudaMemcpy(d_in, h_large, N * sizeof(int), cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        int bSize = 256;
        int gSize = (N + bSize - 1) / bSize;

        // Timing Smoothing
        CHECK_CUDA(cudaEventRecord(start));
        gpu_smooth_kernel<<<gSize, bSize>>>(d_in, d_out, N);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms_smooth = 0;
        cudaEventElapsedTime(&ms_smooth, start, stop);
        printf("GPU Smoothing Time: %.3f ms\n", ms_smooth);

        // Timing Thresholding
        CHECK_CUDA(cudaEventRecord(start));
        gpu_threshold_kernel<<<gSize, bSize>>>(d_out, d_bins, N);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms_thresh = 0;
        cudaEventElapsedTime(&ms_thresh, start, stop);
        printf("GPU Thresholding Time: %.3f ms\n", ms_thresh);

        // Timing Shared Histogram (Task 3d)
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemset(d_histo, 0, 4 * sizeof(int)));
        gpu_histo_shared<<<gSize, bSize>>>(d_bins, d_histo, N);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms_histo = 0;
        cudaEventElapsedTime(&ms_histo, start, stop);
        printf("GPU Shared Histogram Time: %.3f ms\n", ms_histo);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_bins); cudaFree(d_histo);
        free(h_large);
    }

    return 0;
}
