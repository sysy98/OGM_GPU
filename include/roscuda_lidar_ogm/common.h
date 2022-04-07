// Include guard
#ifndef COMMON_H
#define COMMON_H

// headers in STL
#include <stdio.h>
#include <stdlib.h>

// headers in CUDA
#include <cuda_runtime_api.h>

//using MACRO to allocate memory inside CUDA kernel
#define NUM_THREADS_MACRO 256 // need to be changed when NUM_THREADS is changed

#define DIVUP(m, n) (((m) - 1 ) / (n) + 1)

#define GPU_CHECK(ans)                                                                                                 \
  {                                                                                                                    \
    GPUAssert((ans), __FILE__, __LINE__);                                                                              \
  }

inline void GPUAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#endif //COMMON_H