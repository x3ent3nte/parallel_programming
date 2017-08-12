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
    float bias;
    float sum;
    float sig;
    float delta;
};

__global__
void initNeurons(Neuron* layer, int layer_size, int seed) {
    int local_index = threadIdx.x;
    int global_index = local_index + (blockDim.x * blockIdx.x);
    if (global_index >= layer_size) {
        return;
    }

    layer[global_index] = Neuron{hashFloat(seed * (global_index + 1)), 0.0f, 0.0f, 0.0f};
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
void feedInput(Neuron* input_layer,  
                float* inputs,
                int input_layer_size) {
    int local_index = threadIdx.x;
    int global_index = local_index + (blockDim.x * blockIdx.x);
    if (global_index >= input_layer_size) {
        return;
    }
    input_layer[global_index].sig = inputs[global_index]; 
}

__global__
void computeLayer(Neuron* layer, 
                    Neuron* prev_layer, 
                    int layer_size, 
                    int prev_layer_size,
                    float* weights,
                    float* sig_x_weights) {
    int local_index = threadIdx.x;
    int global_index = local_index + (blockDim.x * blockIdx.x);
    if (global_index >= layer_size) {
        return;
    }

    Neuron neuron = layer[global_index];
    int offset = global_index * prev_layer_size;
    float sum = neuron.bias;
    for (int i = 0; i < prev_layer_size; i++) {
        float sig_in = prev_layer[i].sig;
        float sig_x_weight = sig_in * weights[offset + i];
        sig_x_weights[offset + i] = sig_x_weight;
        sum += sig_x_weight;
    }
    neuron.sum = sum;
    neuron.sig = sigmoid(sum);
    layer[global_index] = neuron;
}

__global__
void computeErrors(float* errors,
                    Neuron* output_layer, 
                    float* expected, 
                    int output_layer_size) {
    int local_index = threadIdx.x;
    int global_index = local_index + (blockDim.x * blockIdx.x);
    if (global_index >= output_layer_size) {
        return;
    }

    errors[global_index] = expected[global_index] - output_layer[global_index].sum;
}

__global__
void reduceAbsSum(float* nums, int array_size) {
    int local_index = threadIdx.x;
    int global_offset = blockDim.x * blockIdx.x;
    int global_index = local_index + global_offset;
    
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

__global__
void computeOutputDeltas(Neuron* output_layer,
                            float* errors,
                            int output_layer_size) {
    int local_index = threadIdx.x;
    int global_index = local_index + (blockDim.x * blockIdx.x);
    if (global_index >= output_layer_size) {
        return;
    }

    output_layer[global_index].delta = errors[global_index] * sigmoidPrime(output_layer[global_index].sum);
}

__global__
void computeDeltas(Neuron* layer,
                    Neuron* next_layer,
                    float* next_weights,
                    int layer_size,
                    int next_layer_size) {
    int local_index = threadIdx.x;
    int global_index = local_index + (blockDim.x * blockIdx.x);
    if (global_index >= layer_size) {
        return;
    }

    float sum_delta_weights = 0.0f;
    for (int i = 0; i < next_layer_size; i++) {
        float infront_delta = next_layer[i].delta;
        sum_delta_weights += next_weights[(i * layer_size) + global_index] * infront_delta;
    }
    layer[global_index].delta = sigmoidPrime(layer[global_index].sum) * sum_delta_weights;
}

__global__
void adjustWeights(Neuron* layer,
                     int layer_size,
                     int prev_layer_size,
                     float* weights,
                     float* sig_x_weights) {
    int local_index = threadIdx.x;
    int global_index = local_index + (blockDim.x * blockIdx.x);
    if (global_index >= layer_size) {
        return;
    }

    Neuron neuron = layer[global_index];
    neuron.bias += 2.0f * 1.0f * neuron.delta;
    layer[global_index] = neuron;

    int offset = global_index * prev_layer_size;
    for (int i = 0; i < prev_layer_size; i++) {
        //weights[offset + i] += 2 * sig_x_weights[offset + i] * neuron.delta;
        weights[offset + i] += 2 * neuron.sig * neuron.delta;
    }
}

int main() {
    srand((int) time(NULL));
    int seed = rand();
    printf("seed : %d \n", seed);

    int training_size = 1000;
    int input_size = 2;
    int hidden_size = 8;
    int output_size = 1;
    int num_hidden = 1;

    float** h_inputs = (float**) malloc(sizeof(float*) * training_size);
    for (int i = 0; i < training_size; i++) {
        h_inputs[i] = (float*) malloc(sizeof(float) * input_size);
        h_inputs[i][0] = randomFloat();
        h_inputs[i][1] = randomFloat();
    }

    float** h_expected = (float**) malloc(sizeof(float*) * training_size);
    for (int i = 0; i < training_size; i++) {
        h_expected[i] = (float*) malloc(sizeof(float) * output_size);
        h_expected[i][0] = h_inputs[i][0] + h_inputs[i][1];
    }
    
    float** d_inputs = (float**) malloc(sizeof(float*) * training_size);
    for (int i = 0; i < training_size; i++) {
        cudaMalloc(&d_inputs[i], sizeof(float) * input_size);
        cudaMemcpy(d_inputs[i], h_inputs[i], sizeof(float) * input_size, cudaMemcpyHostToDevice);
    }
    
    float** d_expected = (float**) malloc(sizeof(float*) * training_size);
    for (int i = 0; i < training_size; i++) {
        cudaMalloc(&d_expected[i], sizeof(float) * output_size);
        cudaMemcpy(d_expected[i], h_expected[i], sizeof(float) * output_size, cudaMemcpyHostToDevice);
    }

    float* d_errors;
    cudaMalloc(&d_errors, sizeof(float) * output_size); 

    Neuron* d_input_layer;
    cudaMalloc(&d_input_layer, sizeof(Neuron) * input_size);

    Neuron** d_hidden_layers = (Neuron**) malloc(sizeof(Neuron*) * num_hidden);
    for (int i = 0; i < num_hidden; i++) {
        cudaMalloc(&d_hidden_layers[i], sizeof(Neuron) * hidden_size);
    }

    Neuron* d_output_layer;
    cudaMalloc(&d_output_layer, sizeof(Neuron) * output_size);

    float** d_hidden_weights = (float**) malloc(sizeof(float*) * num_hidden);
    cudaMalloc(&d_hidden_weights[0], sizeof(float) * hidden_size * input_size);
    for (int i = 1; i < num_hidden; i++) {
        cudaMalloc(&d_hidden_weights[i], sizeof(float) * hidden_size * hidden_size);
    }

    float** d_hidden_sig_x_weights = (float**) malloc(sizeof(float*) * num_hidden);
    cudaMalloc(&d_hidden_sig_x_weights[0], sizeof(float) * hidden_size * input_size);
    for (int i = 1; i < num_hidden; i++) {
        cudaMalloc(&d_hidden_sig_x_weights[i], sizeof(float) * hidden_size * hidden_size);
    }

    float* d_output_weights;
    cudaMalloc(&d_output_weights, sizeof(float) * output_size * hidden_size);

    float* d_output_sig_x_weights;
    cudaMalloc(&d_output_sig_x_weights, sizeof(float) * output_size * hidden_size);

    initNeurons<<<1, input_size>>>(d_input_layer, input_size, seed);
    for (int i = 0; i < num_hidden; i++) {
        initNeurons<<<1, hidden_size>>>(d_hidden_layers[i], hidden_size, seed + 1 + i);
    }
    initNeurons<<<1, output_size>>>(d_output_layer, output_size, seed + 1 + num_hidden);

    fillRandomFloatBetween01<<<1, hidden_size * input_size>>>(d_hidden_weights[0], hidden_size * input_size, seed);
    for (int i = 1; i < num_hidden; i++) {
        fillRandomFloatBetween01<<<1, hidden_size>>>(d_hidden_weights[i], hidden_size * hidden_size, seed + i);
    }
    fillRandomFloatBetween01<<<1, output_size * hidden_size>>>(d_output_weights, output_size * hidden_size, seed + num_hidden + 10);

    float avg_error = 9999.9f;

    for (int i = 0 ; i < 100; i++) {
        printf("Avg Error: %f \n", avg_error);
        avg_error = 0.0f;
        int num_trials = training_size * 10;
        for (int j = 0; j < num_trials; j++) {
            int data_row = j % training_size;

            feedInput<<<1, input_size>>>(d_input_layer, d_inputs[data_row], input_size);

            computeLayer<<<1, hidden_size>>>(d_hidden_layers[0], d_input_layer, hidden_size, input_size, d_hidden_weights[0], d_hidden_sig_x_weights[0]);
            for (int hidden = 1; hidden < num_hidden; hidden++) {
                computeLayer<<<1, hidden_size>>>(d_hidden_layers[hidden], d_hidden_layers[hidden - 1], hidden_size, hidden_size, d_hidden_weights[hidden], d_hidden_sig_x_weights[hidden]);
            }
            computeLayer<<<1, output_size>>>(d_output_layer, d_hidden_layers[num_hidden - 1], output_size, hidden_size, d_output_weights, d_output_sig_x_weights);

            computeErrors<<<1, output_size>>>(d_errors, d_output_layer, d_expected[data_row], output_size);

            computeOutputDeltas<<<1, output_size>>>(d_output_layer, d_errors, output_size);
            computeDeltas<<<1, hidden_size>>>(d_hidden_layers[num_hidden - 1], d_output_layer, d_output_weights, hidden_size, output_size);
            for (int hidden = num_hidden - 2; hidden >= 0; hidden--) {
                computeDeltas<<<1, hidden_size>>>(d_hidden_layers[hidden], d_hidden_layers[hidden + 1], d_hidden_weights[hidden + 1], hidden_size, hidden_size);
            }

            adjustWeights<<<1, output_size>>>(d_output_layer, output_size, hidden_size, d_output_weights, d_output_sig_x_weights);
            for (int hidden = num_hidden - 1; hidden >= 1; hidden--) {
                adjustWeights<<<1, hidden_size>>>(d_hidden_layers[hidden], hidden_size, hidden_size, d_hidden_weights[hidden], d_hidden_sig_x_weights[hidden]);
            }
            adjustWeights<<<1, hidden_size>>>(d_hidden_layers[0], hidden_size, input_size, d_hidden_weights[0], d_hidden_sig_x_weights[0]);

            reduceAbsSum<<<1, output_size>>>(d_errors, output_size);

            float sum_errors = 0.0f;
            cudaMemcpy(&sum_errors, &d_errors[0], sizeof(float), cudaMemcpyDeviceToHost);
            avg_error += sum_errors;
        }
        avg_error /= num_trials;
    }

}
