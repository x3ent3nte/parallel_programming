#include "reference_calc.cpp"
#include "utils.h"

__device__ 
float myMax(float x, float y)
{
    if(x > y)
    {
        return x;
    }
    else
    {
        return y;
    }
}

__device__ 
float myMin(float x, float y)
{
    if(x < y)
    {
        return x;
    }
    else
    {
        return y;
    }
}

__global__
void setZeroes(unsigned int* const nums, int array_size)
{
    int local_index = threadIdx.x;
    int global_index = (blockIdx.x * blockDim.x) + local_index;
    if(global_index >= array_size)
    {
        return;
    }
    nums[global_index] = 0;
}

__global__
void reduceMaxMin(const float* const logs, float* partial_reduces, int array_size, bool is_max)
{
    extern __shared__ float sh_log[];
    int local_index = threadIdx.x;
    int global_offset = blockDim.x * blockIdx.x;
    int global_index = local_index + global_offset;

    if(global_index >= array_size)
    {
        return;
    }
    
    sh_log[local_index] = logs[global_index];
    __syncthreads();

    for(unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if(local_index < offset)
        {
            int right = local_index + offset;
            if(right + global_offset < array_size)
            {
                if(is_max)
                {
                    sh_log[local_index] = myMax(sh_log[local_index], sh_log[right]);
                }
                else
                {
                    sh_log[local_index] = myMin(sh_log[local_index], sh_log[right]);
                }
            }
        }
        __syncthreads();
    }

    if(local_index == 0)
    {
        partial_reduces[blockIdx.x] = sh_log[local_index];
    }
}

__global__
void fillHistogram(const float* const logs, unsigned int* const histo, float lum_min, float lum_range, int num_bins, int array_size)
{
    int local_index = threadIdx.x;
    int global_index = (blockIdx.x * blockDim.x) + local_index;
    if(global_index >= array_size)
    {
        return;
    }

    float ratio = (logs[global_index] - lum_min) / lum_range;
    int bin_index = ratio * num_bins;
    if(bin_index >= num_bins)
    {
        bin_index -= 1;
    }
    atomicAdd(&histo[bin_index], 1);
}

__global__
void scanExclusiveSum(unsigned int* const nums, unsigned int* const c_nums, int array_size)
{
    extern __shared__ unsigned int sh_nums[];
    int local_index = threadIdx.x;
    int global_index = (blockDim.x * blockIdx.x) + local_index;

    if(global_index >= array_size)
    {
        return;
    }

    sh_nums[local_index] = nums[global_index];
    __syncthreads();

    for(int offset = 1; offset < blockDim.x; offset <<= 1)
    {
        int left = local_index - offset;
        int left_val = 0;
        if(left >= 0)
        {
            left_val = sh_nums[left];
        }
        __syncthreads();
        if(left >= 0)
        {
            sh_nums[local_index] += left_val;
        }
        __syncthreads();
    }

    if(local_index == 0)
    {
        c_nums[global_index] = 0;
    }
    else
    {
        c_nums[global_index] = sh_nums[local_index - 1];
    }
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
    int num_cells = numRows * numCols;
    float max = 0.0f;
    float min = 0.0f;
    
    int num_blocks = ceil(num_cells / 1024.0f);

    setZeroes<<<1, 1024>>>(d_cdf, numBins);

    float* partial_reduces;
    cudaMalloc(&partial_reduces, sizeof(float) * num_blocks);

    reduceMaxMin<<<num_blocks, 1024, sizeof(float) * 1024>>>(d_logLuminance, partial_reduces, num_cells, true);
    while(num_blocks > 1)
    {
        int prev_num_blocks = num_blocks;
        num_blocks = ceil(num_blocks / 1024.0f);
        reduceMaxMin<<<num_blocks, 1024, sizeof(float) * 1024>>>(partial_reduces, partial_reduces, prev_num_blocks, true);
    }
    cudaMemcpy(&max, &partial_reduces[0], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_logLum, &partial_reduces[0], sizeof(float), cudaMemcpyDeviceToHost);
    
    num_blocks = ceil(num_cells / 1024.0f);
    reduceMaxMin<<<num_blocks, 1024, sizeof(float) * 1024>>>(d_logLuminance, partial_reduces, num_cells, false);
    while(num_blocks > 1)
    {
        int prev_num_blocks = num_blocks;
        num_blocks = ceil(num_blocks / 1024.0f);
        reduceMaxMin<<<num_blocks, 1024, sizeof(float) * 1024>>>(partial_reduces, partial_reduces, prev_num_blocks, false);
    }
    cudaMemcpy(&min, &partial_reduces[0], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&min_logLum, &partial_reduces[0], sizeof(float), cudaMemcpyDeviceToHost);

    float range = max - min;
    num_blocks = ceil(num_cells / 1024.0f);
    fillHistogram<<<num_blocks, 1024>>>(d_logLuminance, d_cdf, min, range, numBins, num_cells);
    
    scanExclusiveSum<<<1, 1024, sizeof(unsigned int) * 1024>>>(d_cdf, d_cdf, numBins);
}










