#include "reference_calc.cpp"
#include "utils.h"

__device__
int myMax(int x, int y)
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
int myMin(int x, int y)
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

__device__ 
int coordinates2Index(int col, int row, int numCols, int numRows)
{
    return (row * numCols) + col; 
}

__global__ 
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
    int centre_col = (blockDim.x * blockIdx.x) + threadIdx.x;
    int centre_row = (blockDim.y * blockIdx.y) + threadIdx.y;
    
    if(centre_col >= numCols || centre_row >= numRows)
    {
        return;
    }
    int centre_index = coordinates2Index(centre_col, centre_row, numCols, numRows);
    
    float blurred = 0;
    int offset = filterWidth / 2;
    for(int i = 0; i < filterWidth; i++)
    {
        for(int j = 0; j < filterWidth; j++ )
        {
            int filter_index = (i * filterWidth) + j;
            float blur_value = filter[filter_index];
            
            int filter_col = j - offset;
            int filter_row = i - offset;
            
            int cell_col = centre_col + filter_col;
            int cell_row = centre_row + filter_row;

            cell_col = myMax(cell_col, 0);
            cell_col = myMin(cell_col, numCols - 1);

            cell_row = myMax(cell_row, 0);
            cell_row = myMin(cell_row, numRows - 1);
            
            blurred += blur_value * inputChannel[coordinates2Index(cell_col, cell_row, numCols, numRows)];
        }
    }
    outputChannel[centre_index] = (unsigned char) blurred; 
}

__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
    const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

    if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    {
        return;
    }

    uchar4 cell = inputImageRGBA[thread_1D_pos];
    redChannel[thread_1D_pos] = cell.x;
    greenChannel[thread_1D_pos] = cell.y;
    blueChannel[thread_1D_pos] = cell.z;
}

__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
    const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

    const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

    if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    {
        return;
    }

    unsigned char red   = redChannel[thread_1D_pos];
    unsigned char green = greenChannel[thread_1D_pos];
    unsigned char blue  = blueChannel[thread_1D_pos];

    uchar4 outputPixel = make_uchar4(red, green, blue, 255);

    outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{
    checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));
  
    checkCudaErrors(cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth));
    cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice);
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
    const dim3 blockSize = dim3(32,32,1);
    const dim3 gridSize = dim3(ceil(numCols / 32.0f), ceil(numRows / 32.0f),1);

    separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      
    gaussian_blur<<<gridSize, blockSize>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
    gaussian_blur<<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
    gaussian_blur<<<gridSize, blockSize>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                                 d_greenBlurred,
                                                 d_blueBlurred,
                                                 d_outputImageRGBA,
                                                 numRows,
                                                 numCols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void cleanup() 
{
    checkCudaErrors(cudaFree(d_red));
    checkCudaErrors(cudaFree(d_green));
    checkCudaErrors(cudaFree(d_blue));
    checkCudaErrors(cudaFree(d_filter));
}









