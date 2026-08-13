#include <iostream>

#include "performance_benchmark.h"

using namespace std;

int counter =0 ;

int main(){

	PerformanceBenchmark::benchmarkMatrixMultiplication(512);

	Matrix A(4, 4);
	A.xavierInit();
	A.print("Matrix A:");

	cout<<"Matrix multiplication benchmark completed."<<endl;

	PerformanceBenchmark::benchmarkNN();

	return 0;
}
