// NOTE(karlluo): compile with: nvcc -arch=sm_90 -O3 code.cu -o demo

#include "cuda_runtime.h"
#include <stdint.h>

constexpr int BLOCK_SIZE = 32;

// NOTE(karlluo): for confidential reason, this is a mimic of rmsnorm
// for y = norm(x, weight1)
// naive implementation adapted from
// https://github.com/erfanzar/EasyDeL/issues/200
__global__ void mimic_rmsnorm(const float *__restrict__ x,
                              const float *__restrict__ weight1,
                              float *__restrict__ inv_rms,
                              float *__restrict__ y, int32_t seq_len,
                              int32_t hidden_size,
                              const float epsilon = 1e-6f) {
  __shared__ float sdata[1024];
  const int seq_idx = blockIdx.x;
  float sum_squared = 0.0f;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hidden_size;
       i += blockDim.x * gridDim.x) {
    float val = x[seq_idx * hidden_size + i];
    sum_squared += val * val;
  }
  cudaTriggerProgrammaticLaunchCompletion();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float mean_squared = sdata[0] / hidden_size;
    float rms = sqrt(mean_squared + epsilon);
    inv_rms[seq_idx] = 1.0f / rms;
  }
  __syncthreads();

  float rms_inv = inv_rms[seq_idx];
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    y[seq_idx * hidden_size + i] =
        x[seq_idx * hidden_size + i] * rms_inv * weight1[i];
  }
}

// NOTE(karlluo): for confidential reason, this is a mimic of project
// for z = matmul(y, weight2)
// naive implementation adapted from
template <const uint32_t BLOCKSIZE>
__global__ void
mimic_matmul(const int32_t seq_len, const int32_t output_hidden_size,
             const int32_t input_hidden_size, const float *__restrict__ y,
             const float *__restrict__ weight2, float *__restrict__ z) {
  const uint32_t c_row = blockIdx.x;
  const uint32_t c_col = blockIdx.y;
  __shared__ float s_a[BLOCKSIZE * BLOCKSIZE];
  __shared__ float s_w[BLOCKSIZE * BLOCKSIZE];
  const uint32_t thread_col = threadIdx.x % BLOCKSIZE;
  const uint32_t thread_row = threadIdx.x / BLOCKSIZE;
  y += c_row * BLOCKSIZE * input_hidden_size;
  weight2 += c_col * BLOCKSIZE;
  z += c_row * BLOCKSIZE * output_hidden_size + c_col * BLOCKSIZE;
  float tmp = 0.0;

  // NOTE(karlluo): prefetch weight2
  for (int bkIdx = 0; bkIdx < input_hidden_size; bkIdx += BLOCKSIZE) {
    s_w[thread_row * BLOCKSIZE + thread_col] =
        weight2[thread_row * output_hidden_size + thread_col];
    __syncthreads();
    weight2 += BLOCKSIZE * output_hidden_size;
  }
  // NOTE(karlluo): finished prefetch weight2, waiting y finish norm as input
  cudaGridDependencySynchronize();

  for (int bkIdx = 0; bkIdx < input_hidden_size; bkIdx += BLOCKSIZE) {
    s_a[thread_row * BLOCKSIZE + thread_col] =
        y[thread_row * input_hidden_size + thread_col];
    __syncthreads();
    y += BLOCKSIZE;
    for (int idx = 0; idx < BLOCKSIZE; ++idx) {
      tmp +=
          s_a[thread_row * BLOCKSIZE + idx] * s_w[idx * BLOCKSIZE + thread_col];
    }
    __syncthreads();
  }
  z[thread_row * output_hidden_size + thread_col] = tmp;
}

int main() {
  // y = norm(x, weight1)
  // z = matmul(y, weight2)

  const int32_t seq_len = 32;
  const int32_t hidden_size = 512;
  const int32_t output_hidden_size = 128;
  const float epsilon = 1e-6f;
  float *x, *weight1, *inv_rms, *y, *weight2, *z;
  // NOTE(karlluo): for demo, skip data initialization
  cudaMalloc(&x, seq_len * hidden_size * sizeof(float));
  cudaMalloc(&weight1, hidden_size * sizeof(float));
  cudaMalloc(&inv_rms, seq_len * sizeof(float));
  cudaMalloc(&y, seq_len * hidden_size * sizeof(float));
  cudaMalloc(&weight2, hidden_size * output_hidden_size * sizeof(float));
  cudaMalloc(&z, seq_len * output_hidden_size * sizeof(float));

  const int32_t norm_grid = seq_len;
  const int32_t norm_block = std::min(hidden_size, 1024);
  dim3 matmul_grid((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 matmul_block(BLOCK_SIZE * BLOCK_SIZE);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  for (int i = 0; i < 10; i++) {
    cudaLaunchConfig_t kernelConfig2 = {0};
    kernelConfig2.gridDim = matmul_grid;
    kernelConfig2.blockDim = matmul_block;
    kernelConfig2.dynamicSmemBytes = 0;
    kernelConfig2.stream = stream;
    cudaLaunchAttribute attribute2[1];
    attribute2[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute2[0].val.programmaticStreamSerializationAllowed = 1;
    kernelConfig2.attrs = attribute2;
    kernelConfig2.numAttrs = 1;

    // PDL Launch
    // mimic_rmsnorm<<<norm_grid, norm_block, 0, stream>>>(
    //     x, weight1, inv_rms, y, seq_len, hidden_size, epsilon);
    // cudaLaunchKernelEx(&kernelConfig2, &(mimic_matmul<BLOCK_SIZE>), seq_len,
    //                    output_hidden_size, hidden_size, y, weight2, z);

    // No PDL Launch
    mimic_rmsnorm<<<norm_grid, norm_block, 0, stream>>>(
        x, weight1, inv_rms, y, seq_len, hidden_size, epsilon);
    mimic_matmul<BLOCK_SIZE><<<matmul_grid, matmul_block, 0, stream>>>(
        seq_len, output_hidden_size, hidden_size, y, weight2, z);
  }

  cudaDeviceSynchronize();

  cudaFree(x);
  cudaFree(weight1);
  cudaFree(inv_rms);
  cudaFree(y);
  cudaFree(weight2);
  cudaFree(z);

  cudaStreamDestroy(stream);
}