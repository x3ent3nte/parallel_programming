#include <stdio.h>
#include <random>

#define E 2.71828182845904523536
#define EPSILON 0.0005

#define INPUT_SIZE 2
#define HIDDEN_SIZE 10
#define OUTPUT_SIZE 1
#define NUM_HIDDEN_LAYERS 4

#define TRAIN_SET_SIZE 100

__device__ 
float sigmoid(float x) {
    return 1.0 / (1.0 + powf(E, -x)); 
}

__device__
float sigmoidPrime(float x) {
    float sig = sigmoid(x);
    return sig * (1 - sig);
}

struct Neuron {
    float2 weights_signal[HIDDEN_SIZE];
    float bias;

    float input;
    float signal; 
    
    float delta;
}

float randomFloat0to1() {
    return ((float) rand()) / ((float) RAND_MAX);
}

Neuron createNeuron() {
    float2 weights_signal[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        weights_signal[i] = float2(randomFloat0to1(), 0.0);
    }
    return Neuron{
        weights_signal,
        1.0,
        0.0,
        0.0,
        0.0
    };
}

Neuron* createNeuronLayer(int size) {
    Neuron* layer = (Neuron*) malloc(sizeof(Neuron) * size);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        layer[i] = createNeuron();
    }
    return layer;
}

__kernel__
void feedInput(Neuron* input_layer, float* inputs[INPUT_SIZE], int data_point, int input_layer_size) {
    int local_index = threadIdx.x;
    int global_index = local_index + (blockDim.x * blockIdx.x);

    if (global_index >= layer_size) {
        return;
    }

    float input = inputs[data_point][global_index];
    input_layer[i].signal = input;
}

__kernel__
void computeLayer(Neuron* layer, Neuron* prev_layer, int layer_size, int prev_layer_size) {
    int local_index = threadIdx.x;
    int global_index = local_index + (blockDim.x * blockIdx.x);

    if (global_index >= layer_size) {
        return;
    }

    Neuron neuron = layer[global_index];
    float sum = neuron.bias;
    for (int i = 0; i < prev_layer_size; i++) {
        float input = prev_layer[i].signal;
        float weight_x_signal = input * neuron.weights_signal[i].x;
        neuron.weights[i].y = weight_x_signal;
        sum += weight_x_signal;
    }
    neuron.input = sum;
    neuron.signal = sigmoid(sum);
    layer[global_index] = neuron;
}

__kernel__
void computeErrors(Neuron* output_layer, float* expected[OUTPUT_SIZE], float* errors, int data_point, int output_layer_size) {
    int local_index = threadIdx.x;
    int global_index = local_index + (blockDim.x * blockIdx.x);

    if (global_index > output_layer_size) {
        return;
    }
    errors[global_index] = expected[data_point][global_index] - output_layer[global_index].input;
}

__kernel__
void reduceAbsSum(float* nums, int array_size) {
    int local_index = threadIdx.x;
    int global_index = local_index + (blockIdx.x * blockDim.x);
    if (global_index >= array_size) {
        return;
    }

    nums[local_index] = abs(nums[local_index]);
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (local_index < offset) {
            int right = local_index + offset;
            if (right + global_offset < array_size) {
                nums[local_index] += nums[right];
            }
        }
        __syncthreads();
    }
}

__kernel__
void computeOutputDeltas(float error, Neuron* output_layer, int output_layer_size) {
    int local_index = threadIdx.x;
    int global_offset = blockDim.x * blockIdx.x;
    int global_index = local_index + global_offset;

    if (global_index >= output_layer_size) {
        return;
    }

    Neuron neuron = output_layer[global_index];
    neuron.delta = error * sigmoidPrime(neuron.input);
}

__kernel__
void computeDeltas(Neuron* layer, Neuron* next_layer, int layer_size, int next_layer_size) {
    int local_index = threadIdx.x;
    int global_index = local_index + (blockDim.x * blockIdx.x);

    if (global_index >= layer_size) {
        return;
    }

    float sum_delta_weights = 0.0;
    for (int i = 0; i < next_layer_size; i++) {
        Neuron infront = next_layer[i];
        sum_delta_weights += infront.weights_signal[global_index].y * infront.delta;
    }
    layer[global_index].delta = sigmoidPrime(sum_delta_weights);
}

__kernel__
void adjustWeights(Neuron* layer, int layer_size, int prev_layer_size) {
    int local_index = threadIdx.x;
    int global_index = local_index + (blockDim.x * blockIdx.x);

    if (global_index >= layer_size) {
        return;
    }

    Neuron neuron = layer[global_index];
    neuron.bias += 2 * 1 * neuron.delta;
    for (int i = 0; i < prev_layer_size; i++) {
        neuron.weights_signal[i].x += 2 * neuron.weights_signal[i].y * neuron.delta; 
    }
    layer[global_index] = neuron;
}

int main() {
    srand((int) time(NULL));

    Neuron* h_input_layer = createNeuronLayer(INPUT_SIZE);
    Neuron* h_hidden_layers[NUM_HIDDEN_LAYERS];
    for(int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        h_hidden_layers[i] = createNeuronLayer[HIDDEN_SIZE];
    }
    Neuron* h_output_layer = createNeuronLayer(OUTPUT_SIZE);

    Neuron* d_input_layer;
    Neuron* d_hidden_layers[NUM_HIDDEN_LAYERS];
    Neuron* d_output_layer;

    cudaMalloc(&d_input_layer, sizeof(Neuron) * INPUT_SIZE);
    cudaMalloc(&d_hidden_layers, sizeof(NEURON) * HIDDEN_SIZE * NUM_HIDDEN_LAYERS);
    cudaMalloc(&d_output_layer, sizeof(Neuron) * OUTPUT_SIZE);

    cudaMemcpy(d_input_layer, h_input_layer, sizeof(Neuron) * INPUT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hidden_layers, h_hidden_layers, sizeof(Neuron) * HIDDEN_SIZE * NUM_HIDDEN_LAYERS);
    cudaMemcpy(d_output_layer, h_output_layer, sizeof(Neuron) * OUTPUT_SIZE, cudaMemcpyHostToDevice);

    float* h_input[INPUT_SIZE] = (float*) malloc(sizeof(float) * INPUT_SIZE * TRAIN_SET_SIZE);
    float* d_input[INPUT_SIZE];
    cudaMalloc(&d_input, sizeof(float) * INPUT_SIZE * TRAIN_SET_SIZE);

    for (int i = 0; i < TRAIN_SET_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            h_input[i][j] = randomFloat0to1();
        }
    }
    cudaMemcpy(d_input, h_input, sizeof(float) * INPUT_SIZE * TRAIN_SET_SIZE, cudaMemcpyHostToDevice);

    float* h_expected[OUTPUT_SIZE] = (float*) malloc(sizeof(float) * OUTPUT_SIZE * TRAIN_SET_SIZE);
    float* d_expected[OUTPUT_SIZE];
    cudaMalloc(&d_expected, sizeof(float) * OUTPUT_SIZE * TRAIN_SET_SIZE);

    for (int i = 0; i < TRAIN_SET_SIZE; i++) {
        float sum = 0.0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += h_input[i][j];
        }
        h_expected[i] = sum;
    }
    cudaMemcpy(d_expected, h_expected, sizeof(float) * OUTPUT_SIZE * TRAIN_SET_SIZE, cudaMemcpyHostToDevice);

    float* d_errors;
    cudaMalloc(&d_errors, sizeof(float) * OUTPUT_SIZE);

    float avg_error = 999.9;

    for(int i = 0; i < 100; i++) {
        printf("Average Error: %f", avg_error);
        avg_error = 0.0;
        for (int j = 0; j < 10 * TRAIN_SET_SIZE; j++) {
            int data_point = j % TRAIN_SET_SIZE;
            feedInput<<<1, 2>>>(d_input_layer, d_input, data_point, INPUT_SIZE);

            computeLayer<<<1, HIDDEN_SIZE>>>(d_hidden_layers[0], d_input_layer, HIDDEN_SIZE, HIDDEN_SIZE);
            for (int layer_num = 1; layer_num < NUM_HIDDEN_LAYERS; layer_num++) {
                computeLayer<<<1, HIDDEN_SIZE>>>(d_hidden_layers[layer_num], d_hidden_layers[layer_num - 1], HIDDEN_SIZE, HIDDEN_SIZE);
            }
            computeLayer<<<1, HIDDEN_SIZE>>>(d_output_layer, d_hidden_layers[NUM_HIDDEN_LAYERS - 1], OUTPUT_SIZE, HIDDEN_SIZE);

            computeErrors<<<1, OUTPUT_SIZE>>>(d_output_layer, d_expected, d_errors, data_point, OUTPUT_SIZE);
            computeOutputDeltas<<<1, OUTPUT_SIZE>>>(d_output_layer, d_errors, OUTPUT_SIZE);

            computeDeltas<<<1, HIDDEN_SIZE>>>(d_hidden_layers[layer_num - 1], d_output_layer, HIDDEN_SIZE, OUTPUT_SIZE);
            for (int layer_num = layer_num - 2; layer_num >= 0; layer_num--) {
                computeDeltas<<<1, HIDDEN_SIZE>>>(d_hidden_layers[layer_num], d_hidden_layers[layer_num + 1, HIDDEN_SIZE, HIDDEN_SIZE]);
            }

            adjustWeights<<<1, HIDDEN_SIZE>>>(d_output_layer, OUTPUT_SIZE, HIDDEN_SIZE);
            for (int layer_num = NUM_HIDDEN_LAYERS - 1; i >= 0; layer_num--) {
                adjustWeights<<<1, HIDDEN_SIZE>>>(d_hidden_layers[layer_num], HIDDEN_SIZE, HIDDEN_SIZE);
            }

            reduceAbsSum<<<1, OUTPUT_SIZE>>>(d_errors, OUTPUT_SIZE);
            float sum_errors = 0.0;
            cudaMemcpy(&sum_errors, d_errors[0], sizeof(float), cudaMemcpyDeviceToHost);
            avg_error += sum_errors;
        }
        avg_error /= 10 * TRAIN_SET_SIZE;
    }
}




















