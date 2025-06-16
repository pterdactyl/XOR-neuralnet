#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid for backpropagation
double sigmoid_derivative(double x) {
    return x * (1 - x);
}

int main() {
    srand(time(0));

    // XOR training data: inputs and expected outputs
    vector<vector<double>> inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    vector<double> outputs = {0, 1, 1, 0};

    // Initialize weights randomly for 2 input neurons to 2 hidden neurons
    double weights_input_hidden[2][2];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            weights_input_hidden[i][j] = ((double)rand() / RAND_MAX);
        }
    }

    // Weights from hidden neurons to output neuron
    double weights_hidden_output[2];
    for (int i = 0; i < 2; ++i) {
        weights_hidden_output[i] = ((double)rand() / RAND_MAX);
    }

    double learning_rate = 0.5;

    // Training loop
    for (int epoch = 0; epoch < 10000; epoch++) {
        double total_error = 0;

        for (int i = 0; i < inputs.size(); i++) {
            // Forward pass

            // Hidden layer
            double hidden_layer[2];
            for (int j = 0; j < 2; j++) {
                hidden_layer[j] = 0;
                for (int k = 0; k < 2; k++) {
                    hidden_layer[j] += inputs[i][k] * weights_input_hidden[k][j];
                }
                hidden_layer[j] = sigmoid(hidden_layer[j]);
            }

            // Output layer
            double output = 0;
            for (int j = 0; j < 2; j++) {
                output += hidden_layer[j] * weights_hidden_output[j];
            }
            output = sigmoid(output);

            // Calculate error
            double error = outputs[i] - output;
            total_error += error * error;

            // Backpropagation

            // Output delta
            double delta_output = error * sigmoid_derivative(output);

            // Hidden layer deltas
            double delta_hidden[2];
            for (int j = 0; j < 2; j++) {
                delta_hidden[j] = delta_output * weights_hidden_output[j] * sigmoid_derivative(hidden_layer[j]);
            }

            // Update weights hidden to output
            for (int j = 0; j < 2; j++) {
                weights_hidden_output[j] += learning_rate * delta_output * hidden_layer[j];
            }

            // Update weights input to hidden
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    weights_input_hidden[k][j] += learning_rate * delta_hidden[j] * inputs[i][k];
                }
            }
        }

        if (epoch % 1000 == 0) {
            cout << "Epoch " << epoch << ", Error: " << total_error << endl;
        }
    }

    // Test the trained network
    cout << "\nTesting trained network:\n";
    for (int i = 0; i < inputs.size(); i++) {
        double hidden_layer[2];
        for (int j = 0; j < 2; j++) {
            hidden_layer[j] = 0;
            for (int k = 0; k < 2; k++) {
                hidden_layer[j] += inputs[i][k] * weights_input_hidden[k][j];
            }
            hidden_layer[j] = sigmoid(hidden_layer[j]);
        }

        double output = 0;
        for (int j = 0; j < 2; j++) {
            output += hidden_layer[j] * weights_hidden_output[j];
        }
        output = sigmoid(output);

        cout << inputs[i][0] << " XOR " << inputs[i][1] << " = " << output << endl;
    }

    return 0;
}