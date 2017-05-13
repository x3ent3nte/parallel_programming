#include "reference_calc.cpp"
#include "utils.h"

__global__
void setZeroes(unsigned int* const nums, int array_size)
{
    int local_index = threadIdx.x;
    int global_index = local_index + (blockDim.x * blockIdx.x);

    if(global_index >= array_size)
    {
        return;
    }

    nums[global_index] = 0;
}

__global__
void shiftRight(unsigned int* const nums, int shift, int array_size)
{
    int local_index = threadIdx.x;
    int global_index = local_index + (blockDim.x * blockIdx.x);

    if(global_index >= array_size)
    {
        return;
    }

    int value_to_write = 0;
    if(global_index > 0)
    {
        value_to_write = nums[global_index - 1];
    }

    __syncthreads();

    nums[global_index] = value_to_write;
}

__global__
void fillHistogram(unsigned int* const nums, unsigned int* histo, int bit_pos, int array_size)
{
    int local_index = threadIdx.x;
    int global_index = local_index + (blockIdx.x * blockDim.x);
    if(global_index >= array_size)
    {
        return;
    }

    int mask = 1 << bit_pos;

    if((nums[global_index] & mask) == 0)
    {
        atomicAdd(&histo[0], 1);
    }
    else
    {
        atomicAdd(&histo[1], 1);
    }
}

__global__
void markFlags(unsigned int* const nums, unsigned int* flags, int bit_pos, bool ones, int array_size)
{
    int local_index = threadIdx.x;
    int global_index = local_index + (blockDim.x * blockIdx.x);

    if(global_index >= array_size)
    {
        return;
    }

    int mask = 1 << bit_pos;
    if(ones)
    {
        if((nums[global_index] & mask) != 0)
        {
            flags[global_index] = 1;
        }
        else
        {
            flags[global_index] = 0;
        }
    }
    else
    {
        if((nums[global_index] & mask) == 0)
        {
            flags[global_index] = 1;
        }
        else
        {
            flags[global_index] = 0;
        }
    }
}

__global__
void scanEnclusiveSumWithBlockCounts(unsigned int* const nums, unsigned int* const c_nums, unsigned int* const block_offsets, int array_size)
{
    extern __shared__ unsigned int sh_nums[];

    int local_index = threadIdx.x;
    int global_index = local_index + (blockDim.x * blockIdx.x);

    if(global_index >= array_size)
    {
        return;
    }

    sh_nums[local_index] = nums[global_index];
    syncthreads();

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

    if(local_index == (blockDim.x - 1))
    {
        block_offsets[blockIdx.x] = sh_nums[local_index];
    }
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

__global__
void combineScanBlocks(unsigned int* const c_nums, unsigned int* const block_offsets, int array_size)
{
    if(blockIdx.x == 0)
    {
        return;
    }   

    int local_index = threadIdx.x;
    int global_index = local_index + (blockDim.x * blockIdx.x);

    if(global_index >= array_size)
    {
        return;
    }

    c_nums[global_index] += block_offsets[blockIdx.x];
}

__global__
void scatterAddresses(unsigned int* const vals, unsigned int* const pos, unsigned int* const out_vals, unsigned int* const out_pos, unsigned int* flags, unsigned int* addresses, int offset, int array_size)
{
    int local_index = threadIdx.x;
    int global_index = local_index + (blockIdx.x * blockDim.x);

    if(global_index >= array_size)
    {
        return;
    }

    if(flags[global_index] == 1)
    {
        int addr = addresses[global_index] + offset;
        out_vals[addr] = vals[global_index];
        out_pos[addr] = pos[global_index];
    }
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    unsigned int* histo;
    cudaMalloc(&histo, sizeof(unsigned int) * 2);

    unsigned int* flags;
    cudaMalloc(&flags, sizeof(unsigned int) * numElems);

    unsigned int* addresses;
    cudaMalloc(&addresses, sizeof(unsigned int) * numElems);

    int num_blocks = ceil(numElems / 1024.0f);

    unsigned int* block_offsets;
    cudaMalloc(&block_offsets, sizeof(unsigned int) * num_blocks);

    unsigned int* first_vals = d_inputVals;
    unsigned int* first_pos = d_inputPos;

    unsigned int* second_vals = d_outputVals;
    unsigned int* second_pos = d_outputPos;

    for(int bit_pos = 0; bit_pos < 31; bit_pos++)
    {
        setZeroes<<<1,2>>>(histo, 2);
        fillHistogram<<<num_blocks, 1024>>>(first_vals, histo, bit_pos, numElems);

        markFlags<<<num_blocks, 1024>>>(first_vals, flags, bit_pos, false, numElems);
        scanEnclusiveSumWithBlockCounts<<<num_blocks, 1024, sizeof(unsigned int) * 1024>>>(flags, addresses, block_offsets, numElems);
        scanExclusiveSum<<<1, num_blocks, sizeof(unsigned int) * 1024>>>(block_offsets, block_offsets, num_blocks);
        combineScanBlocks<<<num_blocks, 1024>>>(addresses, block_offsets, numElems);
        scatterAddresses<<<num_blocks, 1024>>>(first_vals, first_pos, second_vals, second_pos, flags, addresses, 0, numElems);
    
        markFlags<<<num_blocks, 1024>>>(first_vals, flags, bit_pos, true, numElems);
        scanEnclusiveSumWithBlockCounts<<<num_blocks, 1024, sizeof(unsigned int) * 1024>>>(flags, addresses, block_offsets, numElems);
        scanExclusiveSum<<<1, num_blocks, sizeof(unsigned int) * 1024>>>(block_offsets, block_offsets, num_blocks);
        combineScanBlocks<<<num_blocks, 1024>>>(addresses, block_offsets, numElems);

        int offset = 0;
        cudaMemcpy(&offset, &histo[0], sizeof(int), cudaMemcpyDeviceToHost);
        scatterAddresses<<<num_blocks, 1024>>>(first_vals, first_pos, second_vals, second_pos, flags, addresses, offset, numElems);
    
        unsigned int* temp = first_vals;
        first_vals = second_vals;
        second_vals = temp;

        temp = first_pos;
        first_pos = second_pos;
        second_pos = temp;
    }

    cudaFree(histo);
    cudaFree(flags);
    cudaFree(addresses);
    cudaFree(block_offsets);
}



















