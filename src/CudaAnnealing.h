/*
 * CudaAnnealing.h
 *
 *  Created on: Feb 6, 2019
 *      Author: alexander
 */

#ifndef CUDAOPERATIONS_H_
#define CUDAOPERATIONS_H_

#include "Matrice.h"
#include "Spinset.h"

class CudaAnnealing {
private:
	//GPU pointers
	float* devSpins; //Set
	float* devMat; //Mat
	int* devUnemptyMat; //UnemptyMat field of Matrix object
	float* devTemp; //Temperature
	float* meanFieldMembers; //Temporary storage for mean-field computation
	bool* continueIteration;
	double* hamiltonianMembers; //Temporary storage for hamiltonian computation
	//CPU variables
	int size;
	int blockSize;
	int blockCount;
	float minDiff;
public:
	CudaAnnealing(Matrice matrix, int blockCount, float _minDiff);
	void loadSet(Spinset spinset, int index);
	void anneal(float pStep, float linearCoef);
	double extractHamiltonian(int index);
	Spinset extractSet(int index);
	void freeAllocatedMemory();
};

#endif /* CUDAOPERATIONS_H_ */
