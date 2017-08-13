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
};

__global__
void initNeurons(Neuron* layer, int layer_size) {
    int local_id = threadIdx.x;
    int global_id = local_id + (blockDim.x * blockIdx.x);
    if (global_id >= layer_size) {
        return;
    }
    layer[global_id] = Neuron{0.0f, 0.0f, 0.0f};
}

__global__
void fillRandomFloatBetween01(float* nums, int array_size, int seed) {
    int local_id = threadIdx.x;
    int global_id = local_id + (blockDim.x * blockIdx.x);
    if (global_id >= array_size) {
        return;
    }
    nums[global_id] = hashFloat(seed * (global_id + 1));  
}

__global__
void testNetwork(Neuron* neurons,
					float* weights,
					float* results,
					int* layer_sizes,
					int num_layers,
					float* data,
					int num_rows) {
	int local_id = threadIdx.x;

	int input_size = layer_sizes[0];
	int output_size = layer_sizes[num_layers - 1];

	for (int data_row = 0; data_row < num_rows; data_row++) {

		//INPUT DATA
		if (local_id < input_size) {
			neurons[local_id].sig = data[(input_size * data_row) + local_id];
		}
		__syncthreads();

		//FEED FORWARD
		int prev_neuron_offset = 0;
		int current_neuron_offset = input_size;
		int weight_offset = 0;
		for (int i = 1; i < num_layers; i++) {
			if (local_id < layer_sizes[i]) {
				float sum = 0.0f;
				int weights_start = weight_offset + ((layer_sizes[i - 1] + 1) * local_id);
				for (int j = 0; j < layer_sizes[i - 1]; j++) {
					sum += neurons[prev_neuron_offset + j].sig * weights[weights_start + j];
				}
				sum += weights[weights_start + layer_sizes[i - 1]];
				float sig = sigmoid(sum);

				Neuron neuron = neurons[current_neuron_offset + local_id];
				neuron.sum = sum;
				neuron.sig = sig;
				neurons[current_neuron_offset + local_id] = neuron;
			}
			prev_neuron_offset = current_neuron_offset;
			current_neuron_offset += layer_sizes[i];
			weight_offset += ((layer_sizes[i - 1] + 1) * layer_sizes[i]);
			__syncthreads();
		}

		//WRITE RESULTS
		if (local_id < output_size) {
			int result_offset = data_row * output_size;
			results[result_offset + local_id] = neurons[prev_neuron_offset + local_id].sum;
		}
	}
}

__global__
void trainNetwork(Neuron* neurons, 
					float* weights, 
					int* layer_sizes, 
					int num_layers, 
					float* data, 
					float* expected, 
					int num_rows, 
					float learning_rate) {
	int local_id = threadIdx.x;

	int input_size = layer_sizes[0];
	int output_size = layer_sizes[num_layers - 1];

	for (int trial_num = 0; trial_num < num_rows * 10; trial_num++) {
		int data_row = trial_num % num_rows;
		//INPUT DATA
		if (local_id < input_size) {
			neurons[local_id].sig = data[(input_size * data_row) + local_id];
		}
		__syncthreads();

		//FEED FORWARD
		int prev_neuron_offset = 0;
		int current_neuron_offset = input_size;
		int weight_offset = 0;
		for (int i = 1; i < num_layers; i++) {
			if (local_id < layer_sizes[i]) {
				float sum = 0.0f;
				int weights_start = weight_offset + ((layer_sizes[i - 1] + 1) * local_id);
				for (int j = 0; j < layer_sizes[i - 1]; j++) {
					sum += neurons[prev_neuron_offset + j].sig * weights[weights_start + j];
				}
				sum += weights[weights_start + layer_sizes[i - 1]];
				float sig = sigmoid(sum);

				Neuron neuron = neurons[current_neuron_offset + local_id];
				neuron.sum = sum;
				neuron.sig = sig;
				neurons[current_neuron_offset + local_id] = neuron;
			}
			prev_neuron_offset = current_neuron_offset;
			current_neuron_offset += layer_sizes[i];
			weight_offset += ((layer_sizes[i - 1] + 1) * layer_sizes[i]);
			__syncthreads();
		}
		
		//COMPUTE OUTPUT DELTAS
		if (local_id < output_size) {
			Neuron neuron = neurons[prev_neuron_offset + local_id];
			float error = expected[(output_size * data_row) + local_id] - neuron.sum; 
			neuron.delta = error * sigmoidPrime(neuron.sum);
			neurons[prev_neuron_offset + local_id] = neuron;
		}
		__syncthreads();
		
		//COMPUTE HIDDEN DELTAS
		int next_neuron_offset = prev_neuron_offset;
		current_neuron_offset = prev_neuron_offset - layer_sizes[num_layers - 2];
		int next_weight_offset = weight_offset - ((layer_sizes[(num_layers - 2)] + 1) * layer_sizes[num_layers - 1]);
		for (int i = num_layers - 2; i >= 1; i--) {
			if (local_id < layer_sizes[i]) {
				float sum_delta_x_weights = 0.0f;
				for (int j = 0; j < layer_sizes[i + 1]; j++) {
					sum_delta_x_weights += neurons[next_neuron_offset + j].delta * weights[next_weight_offset + ((j * (layer_sizes[i] + 1)) + local_id)];
				}
				Neuron neuron = neurons[current_neuron_offset + local_id];
				neuron.delta = sigmoidPrime(neuron.sum) * sum_delta_x_weights; 
				neurons[current_neuron_offset + local_id] = neuron;
			}
			next_neuron_offset = current_neuron_offset;
			current_neuron_offset -= layer_sizes[i];
			next_weight_offset -= (layer_sizes[i - 1] + 1) * layer_sizes[i];
			__syncthreads();
		}
		
		//ADJUST WEIGHTS
		weight_offset = 0;
		prev_neuron_offset = 0;
		current_neuron_offset = input_size;

		for (int i = 1; i < num_layers; i++) {
			if (local_id < layer_sizes[i]) {
				int weight_start = weight_offset + ((layer_sizes[i - 1] + 1) *  local_id);
				Neuron neuron = neurons[current_neuron_offset + local_id];
				for (int j = 0; j < layer_sizes[i - 1]; j++) {
					weights[weight_start + j] += learning_rate * neuron.delta * neurons[prev_neuron_offset + j].sig;
				}
				weights[weight_start + layer_sizes[i - 1]] += learning_rate * neuron.delta;
			}
			prev_neuron_offset = current_neuron_offset;
			current_neuron_offset += layer_sizes[i];
			weight_offset += (layer_sizes[i - 1] + 1) * layer_sizes[i];
			__syncthreads();
		}
	}
}

int main() {
	srand((int) time(NULL));
	int seed = rand();

	int training_size = 300;
	int input_size = 3;
	int hidden_size = 5;
	int num_hidden_layers = 3;
	int output_size = 1;
	int num_layers = 2 + num_hidden_layers;
	int total_size = input_size + (num_hidden_layers * hidden_size) + output_size;

	int* h_layer_sizes = (int*) malloc(sizeof(int) * num_layers);
	h_layer_sizes[0] = input_size;
	for (int i = 1; i < num_layers - 1; i++) {
		h_layer_sizes[i] = hidden_size;
	}
	h_layer_sizes[num_layers - 1] = output_size;

	int* d_layer_sizes;
	cudaMalloc(&d_layer_sizes, sizeof(int) * num_layers);
	cudaMemcpy(d_layer_sizes, h_layer_sizes, sizeof(int) * num_layers, cudaMemcpyHostToDevice);

	Neuron* d_neurons;
	cudaMalloc(&d_neurons, sizeof(Neuron) * total_size);

	int num_weights = 0;
	for (int i = 1; i < num_layers; i++) {
		num_weights += (h_layer_sizes[i - 1] + 1) * h_layer_sizes[i];
	}

	float* h_weights = (float*) malloc(sizeof(float) * num_weights);
	for (int i = 0; i < num_weights; i++) {
		h_weights[i] = randomFloat();
	}

	float* d_weights;
	cudaMalloc(&d_weights, sizeof(float) * num_weights);
	cudaMemcpy(d_weights, h_weights, sizeof(float) * num_weights, cudaMemcpyHostToDevice);

	float* h_data = (float*) malloc(sizeof(float) * training_size * input_size);
	for (int i = 0; i < training_size * input_size; i++) {
		h_data[i] = randomFloat();
	}

	float* h_expected = (float*) malloc(sizeof(float) * training_size * output_size);
	for (int i = 0; i < training_size; i++) {
		h_expected[i] = h_data[input_size * i] + h_data[(input_size * i) + 1] - h_data[(input_size * i) + 2];
	}


	float* d_data;
	cudaMalloc(&d_data, sizeof(float) * training_size * input_size);
	cudaMemcpy(d_data, h_data, sizeof(float) * training_size * input_size, cudaMemcpyHostToDevice);

	float* d_expected;
	cudaMalloc(&d_expected, sizeof(float) * training_size * output_size);
	cudaMemcpy(d_expected, h_expected, sizeof(float) * training_size * output_size, cudaMemcpyHostToDevice);

	initNeurons<<<1, total_size>>>(d_neurons, total_size);
	//fillRandomFloatBetween01<<<1, num_weights>>>(d_weights, num_weights, seed);

	for (int i = 0; i < 50; i++) {
		trainNetwork<<<1, hidden_size>>>(d_neurons, d_weights, d_layer_sizes, num_layers, d_data, d_expected, training_size, 2.0f);
	}

	int test_size = 30;
	float* h_test_data = (float*) malloc(sizeof(float) * test_size * input_size);
	for (int i = 0; i < test_size * input_size; i++) {
		h_test_data[i] = randomFloat();
	}

	float* d_test_data;
	cudaMalloc(&d_test_data, sizeof(float) * test_size * input_size);
	cudaMemcpy(d_test_data, h_test_data, sizeof(float) * test_size * input_size, cudaMemcpyHostToDevice);

	float* d_results;
	cudaMalloc(&d_results, sizeof(float) * test_size * output_size);

	testNetwork<<<1, hidden_size>>>(d_neurons, d_weights, d_results, d_layer_sizes, num_layers, d_test_data, test_size);

	float* h_results = (float*) malloc(sizeof(float) * test_size * output_size);
	cudaMemcpy(h_results, d_results, sizeof(float) * test_size * output_size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < test_size; i++) {
		float first = h_test_data[input_size * i];
		float second = h_test_data[(input_size * i) + 1];
		float third = h_test_data[(input_size * i) + 2];
		float actual = first + second - third;
		float result = h_results[i];
		float error = result - actual; 
		printf("%f + %f - %f = %f : %f ERROR: %f \n", first, second, third, actual, result, error);
	}

	free(h_layer_sizes);
	free(h_data);
	free(h_expected);
	free(h_test_data);
	free(h_results);
	cudaFree(d_layer_sizes);
	cudaFree(d_neurons);
	cudaFree(d_weights);
	cudaFree(d_data);
	cudaFree(d_expected);
	cudaFree(d_test_data);
	cudaFree(d_results);
}