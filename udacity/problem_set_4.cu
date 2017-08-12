#include "reference_calc.cpp"
#include "utils.h"

__global__
void flipFlags(unsigned int* const nums, int array_size)
{
    int local_index = threadIdx.x;
    int global_index = local_index + (blockDim.x * blockIdx.x);
    if(global_index >= array_size)
    {
        return;
    }
    nums[global_index] ^= 1;
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

    if(local_index == (blockDim.x - 1) || global_index == (array_size - 1))
    {
        block_offsets[blockIdx.x] = sh_nums[local_index];
    }
}

__global__
void scanInclusiveSum(unsigned int* const nums, unsigned int* const c_nums, int array_size)
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
    c_nums[global_index] = sh_nums[local_index];
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

    if(blockIdx.x == 0)
    {
        return;
    }
    c_nums[global_index] += block_offsets[blockIdx.x - 1];
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

__global__
void setOrderFlag(unsigned int* nums, unsigned int* order_flag, int array_size)
{
    int lid = threadIdx.x;
    int gid = lid + (blockDim.x * blockIdx.x);

    if(gid >= array_size)
    {
        return;
    }

    if(gid == 0)
    {
        return;
    }

    if(nums[gid - 1] > nums[gid])
    {
        order_flag[0] = 1;
    }   
}

bool isSorted(unsigned int* nums, int array_size)
{
    unsigned int num_threads = 1024;
    unsigned int num_blocks = ceil(array_size / (float) num_threads);

    unsigned int h_order_flag = 0;
    unsigned int* d_order_flag;
    cudaMalloc(&d_order_flag, sizeof(unsigned int));
    cudaMemset(d_order_flag, 0, sizeof(unsigned int));
    setOrderFlag<<<num_blocks, num_threads>>>(nums, d_order_flag, array_size);
    cudaMemcpy(&h_order_flag, &d_order_flag[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    return h_order_flag == 0;
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
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

    for(int bit_pos = 0; 32; bit_pos++)
    {
        if(isSorted(first_vals, numElems))
        {
            break;
        }
        markFlags<<<num_blocks, 1024>>>(first_vals, flags, bit_pos, false, numElems);
        scanEnclusiveSumWithBlockCounts<<<num_blocks, 1024, sizeof(unsigned int) * 1024>>>(flags, addresses, block_offsets, numElems);
        scanInclusiveSum<<<1, num_blocks, sizeof(unsigned int) * 1024>>>(block_offsets, block_offsets, num_blocks);
        combineScanBlocks<<<num_blocks, 1024>>>(addresses, block_offsets, numElems);
        scatterAddresses<<<num_blocks, 1024>>>(first_vals, first_pos, second_vals, second_pos, flags, addresses, 0, numElems);

        unsigned int offset = 0;
        cudaMemcpy(&offset, &block_offsets[num_blocks - 1], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
        flipFlags<<<num_blocks, 1024>>>(flags, numElems);
        scanEnclusiveSumWithBlockCounts<<<num_blocks, 1024, sizeof(unsigned int) * 1024>>>(flags, addresses, block_offsets, numElems);
        scanInclusiveSum<<<1, num_blocks, sizeof(unsigned int) * 1024>>>(block_offsets, block_offsets, num_blocks);
        combineScanBlocks<<<num_blocks, 1024>>>(addresses, block_offsets, numElems);
        scatterAddresses<<<num_blocks, 1024>>>(first_vals, first_pos, second_vals, second_pos, flags, addresses, offset, numElems);
    
        unsigned int* temp = first_vals;
        first_vals = second_vals;
        second_vals = temp;

        temp = first_pos;
        first_pos = second_pos;
        second_pos = temp;
    }

    cudaFree(flags);
    cudaFree(addresses);
    cudaFree(block_offsets);
}


