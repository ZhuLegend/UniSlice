#include <cuda_runtime.h>

// Function to reset the CUDA device.
void reset_cuda_device() {
    // No need to check for errors, as this is a best-effort cleanup.
    cudaDeviceReset();
}
