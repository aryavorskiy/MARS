/*
 * Matrix.cpp
 *
 *  Created on: Jan 28, 2019
 *      Author: alexander
 */

#include "Matrix.h"
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>

Matrix::Matrix(int size) {
	this->size = size;
	matrix = new float[size * size];
	nonZeroCells = new int[size * (size + 1)];
	for (int var = 0; var < size * (size + 1); ++var) {
		nonZeroCells[var] = 0;
		if (var < size * size)
			matrix[var] = 0.;
	}
	sum = 0;
}

Matrix::Matrix(ifstream fs) {
	int ss;
	fs >> ss;
	size = ss;
	sum = 0;
	matrix = new float[size * size];
	nonZeroCells = new int[size * (size + 1)];
	for (int var = 0; var < size * (size + 1); ++var) {
		nonZeroCells[var] = 0;
		if (var < size * size)
			matrix[var] = 0.;
	}
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			float cell;
			fs >> cell;
			if (i <= j) {
				matrix[i * size + j] = cell;
				matrix[j * size + i] = cell;
				if (cell != 0.) {
					nonZeroCells[i * (size + 1)]++;
					nonZeroCells[i * (size + 1) + nonZeroCells[i * (size + 1)]] =
							j;
					if (i != j) {
						nonZeroCells[j * (size + 1)]++;
						nonZeroCells[j * (size + 1)
								+ nonZeroCells[j * (size + 1)]] = i;
					}
				}
				sum += cell;
			}
		}
	}
}

void Matrix::Randomize() {
	float f;
	sum = 0;
	for (int var = 0; var < size * (size + 1); ++var) {
		nonZeroCells[var] = 0;
	}
	for (int i = 0; i < size; ++i) {
		for (int j = i + 1; j < size; ++j) {
			f = rand() / (float) RAND_MAX;
			f = f * 2 - 1;
			matrix[i * size + j] = f;
			matrix[j * size + i] = matrix[i * size + j];
			if (f != 0.) {
				nonZeroCells[i * (size + 1)]++;
				nonZeroCells[i * (size + 1) + nonZeroCells[i * (size + 1)]] = j;
				nonZeroCells[j * (size + 1)]++;
				nonZeroCells[j * (size + 1) + nonZeroCells[j * (size + 1)]] = i;
			}
			sum += matrix[i * size + j];
		}
	}
}

int Matrix::getSize() {
	return size;
}

string Matrix::getMatrixText() {
	ostringstream out;
	out << size << "\n";
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			out << matrix[i * size + j] << " ";
		}
		out << "\n";
	}
	return out.str();
}

void Matrix::buildMat(ifstream ifs) {
	sum = 0;
	ifs >> size;
	matrix = new float[size * size];
	nonZeroCells = new int[size * (size + 1)];
	for (int var = 0; var < size * (size + 1); ++var) {
		nonZeroCells[var] = 0;
		if (var < size * size)
			matrix[var] = 0.;
	}
	int i, j, val, edges;
	ifs >> edges;
	for (int var = 0; var < edges; ++var) {
		ifs >> i;
		ifs >> j;
		ifs >> val;
		i -= 1;
		j -= 1;
		matrix[i * size + j] = val;
		matrix[j * size + i] = val;
		if (val != 0.) {
			nonZeroCells[i * (size + 1)]++;
			nonZeroCells[i * (size + 1) + nonZeroCells[i * (size + 1)]] = j;
			nonZeroCells[j * (size + 1)]++;
			nonZeroCells[j * (size + 1) + nonZeroCells[j * (size + 1)]] = i;
		}
		sum += val;
	}
}

float Matrix::getSum() {
	return sum;
}

float* Matrix::getArray() {
	return matrix;
}

int* Matrix::getUnemptyMat() {
	return nonZeroCells;
}
