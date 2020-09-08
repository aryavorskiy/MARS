/*
 * Matrice.h
 *
 *  Created on: Jan 28, 2019
 *      Author: alexander
 */

#ifndef MATRICE_H_
#define MATRICE_H_
#include <iostream>
#include <fstream>
using namespace std;

class Matrix {
private:
	int size;
	float* matrix;
	int* nonZeroCells;
	float sum;
public:
	Matrix(int size);
	Matrix(ifstream fs);
	void buildMat(ifstream ifs);
	void Randomize();
	string getMatrixText();
	int getSize();
	float* getArray();
	float getSum();
	int* getUnemptyMat();
};

#endif /* MATRICE_H_ */
