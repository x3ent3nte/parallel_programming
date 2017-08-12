#include <stdio.h>

#define SIZE 1024

__global__ void markEven(int* nums, int* flags)
{
	if(nums[threadIdx.x] & 1 == 0)
	{
		flags[threadIdx.x] = 1;
	}
	else
	{
		flags[threadIdx.x] = 0;
	}
}

__global__ void scanSum(int* nums, int* c_nums)
{
	extern __shared__ int sh_nums[];

	int index = threadIdx.x;
	if(index > 0)
	{
		sh_nums[index] = nums[index - 1];
	}
	else
	{
		sh_nums[index] = 0;
	}
	__syncthreads();

	int offset = 2;

	while(offset < blockDim.x)
	{
		int left = index - offset;
		if(left >= 0)
		{
			sh_nums[index] += sh_nums[left];
		}
		offset <<= 1;

		__syncthreads();
	}
	c_nums[index] = sh_nums[index];
}

__global__ void scatterAddress(int* nums, int* flags, int* address, int* filtered)
{
	int index = threadIdx.x;
	if(flags[index] == 1)
	{
		filtered[address[index]] = nums[index];
	}
}

int main()
{
	int* nums;
	nums = (int*) malloc(sizeof(int) * SIZE);
	for(int i = 0; i < SIZE; i++)
	{
		nums[i] = i;
	}

	int* d_nums;
	cudaMalloc(&d_nums, sizeof(int) * SIZE);
	int* d_flags;
	cudaMalloc(&d_flags, sizeof(int) * SIZE);
	int* d_address;
	cudaMalloc(&d_address, sizeof(int) * SIZE);

	cudaMemcpy(d_nums, nums, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
	markEven<<<1, SIZE, sizeof(int) * SIZE>>>(d_nums, d_flags);
	scanSum<<<1, SIZE, sizeof(int) * SIZE>>>(d_flags, d_address);

	int* filter_size;
	filter_size = (int*) malloc(sizeof(int));
	cudaMemcpy(filter_size, &d_address[SIZE - 1], sizeof(int), cudaMemcpyDeviceToHost);
	int* filtered;
	filtered = (int*) malloc(sizeof(int) * filter_size[0]);
	int* d_filtered;
	cudaMalloc(&d_filtered, sizeof(int) * filter_size[0]);

	scatterAddress<<<1, SIZE>>>(d_nums, d_flags, d_address, d_filtered);
	cudaMemcpy(filtered, d_filtered, sizeof(int) * filter_size[0], cudaMemcpyDeviceToHost);

	for(int i = 0; i < filter_size[0]; i++)
	{
		printf("%d \n", filtered[i]);
	}
}











