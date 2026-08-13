#ifndef NN_H
#define NN_H

#include <cmath>
#include <vector>
#include "matrix.h"
#include "matrix_operation.h"

class ActivationFunction {
	public:

	static double Sigmoid(double x) {
		return 1.0 / (1.0 + exp(-x));
	}

	static double SigmoidDerivative(double x) {
		double sigmoid = Sigmoid(x);
		return sigmoid * (1.0 - sigmoid);
	}

	static Matrix apply(const Matrix& input, double (*func)(double)) {
		Matrix result(input.getRows(), input.getCols());
		for(size_t i = 0; i < input.getRows(); ++i) {
			for (size_t j = 0; j < input.getCols(); ++j) {
				result(i, j) = func(input(i, j));
			}
		}

		return result;
	}
};

class Layer {
	private:

		Matrix weights;
		Matrix biases;
		Matrix lastInput;
		Matrix lastOutput;
		double (*activationFunc)(double);
		double (*activationDerivative)(double);
		MatrixOptimization optLevel;

	public:
		 Layer(size_t inputSize, size_t outputSize,
		       double (*activationFunc)(double) = ActivationFunction::Sigmoid,
			   double (*activationDerivative)(double) = ActivationFunction::SigmoidDerivative,
			   MatrixOptimization optLevel = MatrixOptimization::NAIVE)
			                                   :weights(outputSize, inputSize),
												biases(outputSize, 1),
												lastInput(1,1), lastOutput(1,1),
												activationFunc(activationFunc),
												activationDerivative(activationDerivative) {
			setOptimization(optLevel);
			weights.xavierInit();
		}

		Matrix forward(const Matrix& input);
		Matrix backward(const Matrix &outputGradient, double learningRate);
		void setOptimization(MatrixOptimization opt) { optLevel = opt; }
};

class NeuralNetwork {
	private:
		vector<Layer> layers;
		MatrixOptimization optLevel;

	public:
		NeuralNetwork(MatrixOptimization opt = NAIVE) : optLevel(opt) {}

		void addLayer(size_t inputSize, size_t outputSize,
					  double (*activationFunc)(double) = ActivationFunction::Sigmoid,
					  double (*activationDerivative)(double) = ActivationFunction::SigmoidDerivative,
					  MatrixOptimization opt = MatrixOptimization::NAIVE) {
			layers.emplace_back(inputSize, outputSize, activationFunc, activationDerivative, opt);
		}

		Matrix predict(const Matrix &input);

		void train(const Matrix &input, const Matrix &target, double learningRate = 0.01);

		double computeLoss(const Matrix &input, const Matrix &target);

		void setOptimization(MatrixOptimization opt);
		void getOptimization() {
			cout<<"Current Optimization Level: " << optLevel << endl;
			auto opt = optLevel;
			for(auto &layer : layers) {
				layer.setOptimization(opt);
			}
		}

};

#endif
