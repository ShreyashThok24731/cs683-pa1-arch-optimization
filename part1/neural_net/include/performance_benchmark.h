#ifndef PERFORMANCE_BENCHMARK_H
#define PERFORMANCE_BENCHMARK_H

#include "matrix.h"
#include "defines.h"
#include "matrix_operation.h"
#include "nn.h"
#include <chrono>

using namespace std;

extern int counter;

class PerformanceBenchmark {
	public:

		static void benchmarkMatrixMultiplication(size_t size = 512) {
			cout << "\n=== Matrix Multiplication Performance Benchmark ===\n";
			cout << "Matrix size: " << size << "x" << size << "\n\n";

			Matrix A(size, size);
			Matrix B(size, size);
			A.initRandom();
			B.initRandom();

			vector<pair<MatrixOptimization, string>> optimizations = {
				{MatrixOptimization::NAIVE, "Basic (ijk)"},
				{MatrixOptimization::REORDERED, "Reordered (ikj)"},
				{MatrixOptimization::UNROLLED, "Unrolled"},
				{MatrixOptimization::TILED, "Tiled/Blocked"},
				{MatrixOptimization::VECTORIZED, "SIMD Vectorized"}
			};

			vector<double> times;

			for (const auto& opt : optimizations) {
				auto start = chrono::high_resolution_clock::now();

				Matrix result = MatrixOperation::MatMul(A, B, opt.first);

				auto end = chrono::high_resolution_clock::now();
				auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

				double time_ms = duration.count();
				times.push_back(time_ms);

				cout << setw(20) << opt.second << ": "
						  << setw(8) << fixed << setprecision(2)
						  << time_ms << " ms\n";
			}

			cout << "\nSpeedup over basic implementation:\n";
			double baseline = times[0];
			for (size_t i = 1; i < optimizations.size(); ++i) {
				double speedup = baseline / times[i];
				cout << setw(20) << optimizations[i].second << ": "
						  << setw(6) << fixed << setprecision(2)
						  << speedup << "x\n";
			}
		}

		static void benchmarkNN() {
			cout << "\n=== Neural Network Training Performance ===\n";

        const size_t batchSize = 512;
        const size_t inputSize = 512;
        const size_t hiddenSize = 512;
        const size_t outputSize = 512;
        const size_t epochs = 1;

        vector<pair<MatrixOptimization, string>> optimizations = {
            {MatrixOptimization::NAIVE, "Basic"},
		    {MatrixOptimization::REORDERED, "Reordered"},
			{MatrixOptimization::UNROLLED, "Unrolled"},
			{MatrixOptimization::TILED, "Tiled"},
			{MatrixOptimization::VECTORIZED, "Vectorized"}
        };

        Matrix input(inputSize, batchSize);
        Matrix target(outputSize, batchSize);
        input.initRandom();
        target.initRandom();

		cout << "Input Matrix:\n";
		cout << "Target Matrix:\n";
		cout << "\nTraining with different optimizations:\n";

        for (const auto& opt : optimizations) {
			counter = 0;
            NeuralNetwork nn(opt.first);
            nn.addLayer(inputSize, hiddenSize, ActivationFunction::Sigmoid, ActivationFunction::SigmoidDerivative, opt.first);
            nn.addLayer(hiddenSize, outputSize, ActivationFunction::Sigmoid, ActivationFunction::SigmoidDerivative, opt.first);

            auto start = chrono::high_resolution_clock::now();

            for (size_t epoch = 0; epoch < epochs; ++epoch) {
                nn.train(input, target, 0.01);
            }

            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

            double loss = nn.computeLoss(input, target);

            cout << setw(15) << opt.second
                      << " - Time: " << setw(6) << duration.count() << " ms"
                      << " - Final Loss: " << setw(8) << fixed << setprecision(6) << loss << "\n"
					  << " - Counter: " << counter << "\n";
        }
    }
};

#endif
