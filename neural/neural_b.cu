#include <stdio.h>
#include <random>

#define E 2.71828182845904523536
#define EPSILON 0.0005

float randomFloat() {
    return ((float) rand()) / ((float) RAND_MAX);
}

__device__
int hashInt(int a) {
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}

__device__
float hashFloat(int a) {
    a = hashInt(a);
    int bound = (1 << 31) - 1;
    a &= bound;
    float b = ((float) a) / ((float) bound);
    return b;
}

__device__
float sigmoid(float x) {
    return 1.0f / (1.0f + powf(E, -x));
}

__device__
float sigmoidPrime(float x) {
    float sig_x = sigmoid(x);
    return sig_x * (1.0f - sig_x);
}

struct Neuron {
    float sum;
    float sig;
    float delta;
}

__global__
void initNeurons(Neuron* layer, int layer_size) {
    int local_index = threadIdx.x;
    int global_index = local_index + (blockDim.x * blockIdx.x);
    if (global_index >= layer_size) {
        return;
    }
    layer[global_index] = Neuron{0.0f, 0.0f, 0.0f};
}

__global__
void fillRandomFloatBetween01(float* nums, int array_size, int seed) {
    int local_index = threadIdx.x;
    int global_index = local_index + (blockDim.x * blockIdx.x);
    if (global_index >= array_size) {
        return;
    }
    nums[global_index] = hashFloat(seed * (global_index + 1));  
}

__global__
void computeNetwork(Neuron* neurons, float* weights, int* layer_sizes, int num_layers, float* data, float* expected, int data_point) {
    int local_index = threadIdx.x;
    int global_index = local_index + (blockIdx.x *  blockDim.x);

    int data_offset = layer_sizes[0] * data_point;

    //INPUT DATA
    if (global_index < layer_sizes[0]) {
        neuron[global_index].sig = data[data_offset + global_index]; 
    }
    __syncthreads();

    //FEED FORWARD
    int prev_neuron_offset = 0;
    int current_neuron_offset = layer_size[0];
    int weights_offset = 0;
    for (int i = 1; i < num_layers; i++) {
        if (global_index < layers_size[i]) {
            int weights_start = weights_offset + (global_index * (layer_sizes[i - 1] + 1));
            float sum = 0.0f;
            for (int j = 0; j < layers_size[i - 1]; j++) {
                sum += neurons[prev_neuron_offset + global_index].sig * weights[weights_start + j];
            }
            sum += 1.0f * weights[weights_start + layers_sizes[i - 1]];
            float sig = sigmoid(sum);

            Neuron neuron = neurons[current_neuron_offset + global_index];
            neuron.sum = sum;
            neuron.sig = sig;
            neurons[current_neuron_offset + global_index] = neuron;
        }
        prev_neuron_offset = current_neuron_offset;
        current_neuron_offset += layer_sizes[i];
        weights_offset += (layers_size[i - 1] + 1) * layers_size[i]; 

        __syncthreads();
    }

    //COMPUTE ERRORS AND OUTPUT DELTA
    if (global_index < layer_sizes[num_layers - 1]) {
        Neuron neuron = neurons[prev_neuron_offset + global_index];
        float error = expected[data_offset + global_index] - neuron.sum;
        neuron.delta = error * sigmoidPrime(neuron.sum);
        neurons[prev_neuron_offset + global_index] = neuron;
    }
    __syncthreads();


    //COMPUTE DELTAS
    int next_neuron_offset = prev_neuron_offset;
    current_neuron_offset = prev_neuron_offset - layer_sizes[num_layers - 2];
    weights_offset -= (layer_sizes[num_layers - 2] + 1) * layers[num_layers - 1];
    for (int i = num_layers - 2; i >= 1; i--) {
        if (global_index < layer_sizes[i]) {
            float sum_delta_weights = 0.0f;
            for (int j = 0; j < layer_sizes[i + 1]) {
                sum_delta_weights = neurons[next_neuron_offset + j].delta * weights[weights_offset + (global_index * (layer_sizes[i] + 1))];
            }
            Neuron neuron = neurons[current_neuron_offset + global_index];
            neuron.delta = neuron.sum * sum_delta_weights;
        }
        next_neuron_offset = current_neuron_offset;
        current_neuron_offset -= layer_sizes[i];
        weights_offset -= (layer_sizes[i - 1] + 1) * layers[i];

        __syncthreads();
    }
    __syncthreads();

    //ADJUST WEIGHTS
    weights_offset = 0;
    current_neuron_offset = layer_sizes[0];
    for (int i = 1; i < num_layers i++) {
        int weights_start = weights_offset + (global_index * (layer_sizes[i - 1] + 1));
        if (global_index < layers[i]) {
            Neuron neuron = neurons[current_neuron_offset + global_index]; 
            for (int j = 0; j < layer_sizes[i - 1]; j++) {
                weights[weights_start + j] += 2 * neuron.sig * neuron.delta;
            }
            weights[weights_start + layer_sizes[i - 1]] += 2.0f * 1.0f * neuron.delta;
        }
        weights_offset += (layer_sizes[i - 1] + 1) * layer_sizes[i];
        current_neuron_offset += layer_sizes[i]; 
        __syncthreads();
    }
    
}

int main() {
    srand((int) time(NULL));
    int seed = rand();

    int training_size = 100;
    int in_size = 2;
    int hidden_size = 8;
    int out_size = 1;

    Neuron* d_neurons;
    cudaMalloc(&d_neurons, sizeof(Neuron) * (in_size + hidden_size + out_size));

    float* d_weights;
    cudaMalloc(&d_weights, sizeof(float) * (((in_size + 1) * hidden_size) + ((hidden_size + 1) * out_size)));

    int* h_layer_sizes = (int*) malloc(sizeof(int) * 3);
    h_layers_sizes[0] = 2;
    h_layers_sizes[1] = 8;
    h_layer_sizes[2] = 1;

    int* d_layer_sizes;
    cudaMalloc(&d_layer_sizes, sizeof(int) * 3);
    cudaMemcpy(d_layer_sizes, h_layer_sizes, sizeof(int) * 3, cudaMemcpyHostToDevice);

    float* h_data = (float*) malloc(sizeof(float) * training_size * in_size);
    for (int i = 0; i < training_size * in_size; i++) {
        h_data[i] = randomFloat();
    }

    float* h_expected = (float*) malloc(sizeof(float) * training_size * out_size);
    for (int i = 0; i < training_size *out_size; i++) {
        h_expected[i] = h_data[2 * i] + h_data[(2 * i) + 1];
    }

    float* d_data;
    cudaMalloc(&d_data, sizeof(float) * training_size * in_size);
    cudaMemcpy(d_data, h_data, sizeof(float) * training_size * in_size, cudaMemcpyHostToDevice);

    float* d_expected;
    cudaMalloc(&d_expected, sizeof(float) * training_size * out_size);
    cudaMemcpy(d_expected, h_expected, sizeof(float) * training_size * out_size, cudaMemcpyHostToDevice);

    computeNetwork<<<1, 512>>>(d_neurons, d_weights, d_layer_sizes, 3, d_data, d_expected, 0);
}






























