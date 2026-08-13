#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <random>

#include "defines.h"

using namespace std;


class Matrix {
	private:
		vector<vector<element_t>> mat;
		size_t rows, cols;
	public:
		Matrix(size_t r, size_t c): rows(r), cols(c) {
			mat.resize(rows, vector<element_t>(cols, INITIAL_VALUE));
		}
		Matrix(const vector<vector<element_t>> data): mat(data) {
			rows = data.size();
			cols = rows > 0 ? data[0].size() : 0;
		}

		element_t& operator()(size_t r, size_t c) { return mat[r][c]; }
		const element_t& operator()(size_t r, size_t c) const { return mat[r][c]; }

		size_t getRows() const { return rows; }
		size_t getCols() const { return cols; }

		void initRandom(double min=-1.0, double max=1.0) {
			random_device rd;
			mt19937 gen(rd());
			uniform_real_distribution<double> dis(min, max);
			for (size_t i = 0; i < rows; ++i) {
				for (size_t j = 0; j < cols; ++j) {
					mat[i][j] = dis(gen);
				}
			}
		}

		void xavierInit() {
			double range = sqrt(6.0 / (rows + cols));
			initRandom(-range, range);
		}

		void print(char* matrix) const {
			cout<<"Matrix: "<<matrix<<endl;
			for (size_t i = 0; i < rows; ++i) {
				for (size_t j = 0; j < cols; ++j) {
					cout << setw(8) << fixed << setprecision(4) << mat[i][j] << " ";
				}
				cout << endl;
			}
		}
};

#endif
