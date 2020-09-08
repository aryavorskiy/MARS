/*
 * CudaAnnealing.cu
 *
 *  Created on: Feb 6, 2019
 *      Author: alexander
 */

#include "Matrix.h"
#include "Spinset.h"
#include "CudaAnnealing.h"
#include <cuda_runtime.h>
#include <sstream>
#include <math.h>

void checkError(cudaError_t err, string arg = "") {
	if (err != cudaSuccess) {
		cout << "Error: " << cudaGetErrorString(err) << endl;
		if (arg != "")
			cout << "Additional data: " << arg << endl;
		std::exit(-1);
	}
}

CudaAnnealing::CudaAnnealing(Matrix _matrix, int _blockCount, float _minDiff) {
	minDiff = _minDiff;
	// Set pointers to null
	devSpins = NULL;
	devMat = NULL;
	devUnemptyMat = NULL;
	meanFieldMembers = NULL;
	hamiltonianMembers = NULL;
	continueIteration = NULL;
	devTemp = NULL;

	size = _matrix.getSize();
	blockSize = 512;
	blockCount = _blockCount;

	cudaDeviceProp deviceProp;
	checkError(cudaGetDeviceProperties(&deviceProp, 0), "getProp");
	blockSize = deviceProp.maxThreadsPerBlock;

	// Allocate memory for pointers at GPU
	checkError(
			cudaMalloc((void**) &meanFieldMembers,
					sizeof(float) * size * blockCount), "malloc");
	cudaMalloc((void**) &devMat, sizeof(float) * size * size);
	cudaMalloc((void**) &devSpins, sizeof(float) * size * blockCount);
	cudaMalloc((void**) &devUnemptyMat, sizeof(int) * size * (size + 1));
	cudaMalloc((void**) &hamiltonianMembers, sizeof(double) * size * size);
	cudaMalloc((void**) &devTemp, sizeof(float) * blockCount);
	cudaMalloc((void**) &continueIteration, sizeof(bool) * _blockCount);

	// Copy model data to GPU memory
	checkError(
			cudaMemcpy(devMat, _matrix.getArray(), sizeof(float) * size * size,
					cudaMemcpyHostToDevice), "memcpy mat to host");
	cudaMemcpy(devUnemptyMat, _matrix.getUnemptyMat(),
			sizeof(int) * size * (size + 1), cudaMemcpyHostToDevice);
}

void CudaAnnealing::loadSet(Spinset set, int setIndex) {
	checkError(
			cudaMemcpy(&devSpins[setIndex * size], set.getArray(),
					sizeof(float) * size, cudaMemcpyHostToDevice),
			"memcpy spinset to device");
	cudaMemcpy(&devTemp[setIndex], &(set.temp), sizeof(float),
			cudaMemcpyHostToDevice);
}

void CudaAnnealing::freeAllocatedMemory() {
	// Free GPU memory
	cudaFree(devSpins);
	cudaFree(devMat);
	cudaFree(meanFieldMembers);
	cudaFree(devTemp);
	cudaFree(devUnemptyMat);
	cudaFree(hamiltonianMembers);
	cudaFree(continueIteration);
}

__global__ void allocateHamiltonianMembers(float* devMat, float* devSpins,
		int setIndex, int size, double* hamiltonianMembers) {
	// Hamiltonian member assignment
	int i;
	int j;

	int wIndex = threadIdx.x + blockIdx.x * blockDim.x;
	while (wIndex < size * size) {
		i = wIndex % size;
		j = (int) (wIndex / size);
		if (i == j)
			hamiltonianMembers[wIndex] = devSpins[i + setIndex * size]
					* devMat[wIndex];
		else if (i > j)
			hamiltonianMembers[wIndex] = (double) (devSpins[i + setIndex * size]
					* devSpins[j + setIndex * size] * devMat[wIndex]);
		else
			hamiltonianMembers[wIndex] = 0;
		wIndex = wIndex + blockDim.x * gridDim.x;
	}
}

__global__ void quickSum(double* members, int size) {
	// Sum up numbers in specified range within specified pointer
	// In the end she sum will be accessible directly from pointer
	long long offset = 1;
	int wIndex;
	while (offset < size) {
		wIndex = threadIdx.x;
		while ((wIndex * 2 + 1) * offset < size) {
			members[wIndex * 2 * offset] += members[(wIndex * 2 + 1) * offset];
			wIndex = wIndex + blockDim.x;
		}
		offset *= 2;
		__syncthreads();
	}
}

double CudaAnnealing::extractHamiltonian(int index) { // Get hamiltonian from set with index
	allocateHamiltonianMembers<<<blockCount, blockSize>>>(devMat, devSpins, index, size,
			hamiltonianMembers);
	quickSum<<<1, blockSize>>>(hamiltonianMembers, size * size);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	checkError(err, "Kernel at extractEnergy");
	double out;
	checkError(
			cudaMemcpy(&out, hamiltonianMembers, sizeof(double),
					cudaMemcpyDeviceToHost), "memcpy energy to host");
	return out;
}

Spinset CudaAnnealing::extractSet(int index) { // Get spins from set with index
	float* hSpins = (float*) malloc(sizeof(float) * size);
	checkError(
			cudaMemcpy(hSpins, &devSpins[index * size], sizeof(float) * size,
					cudaMemcpyDeviceToHost), "memcpy spins to host");
	Spinset outSpins(size);
	for (int i = 0; i < size; i++)
		outSpins.SetSpin(i, hSpins[i]);
	return outSpins;
}

__device__ float meanFieldMember(const float *mat, const float *set,
		int spinIndex, int i, int size) {  // Returns /Phi_ind
	if (i != spinIndex)
		return mat[spinIndex * size + i] * set[i];
	else
		return mat[spinIndex * size + i];
}

__global__ void cudaKernelAnneal(float* mat, float* spins, int size,
		float* temp, float tempStep, float* meanFieldMembers,
		bool* proceedFlags, float proceedThreshold, int* unemptyCells,
		float linearCoef) {
	int blockId = blockIdx.x;
	int thrId = threadIdx.x;

	do {
		// Decrease temperature
		if (thrId == 0)
			temp[blockId] = temp[blockId] - tempStep;

		// Stabilize
		do {
			__syncthreads();
			// Resetting flags
			if (thrId == 0)
				proceedFlags[blockId] = false;

			for (int spinId = 0; spinId < size; ++spinId) { // Anneal every spin
				__syncthreads();

				// Mean-field member assignment
				int wIndex = thrId;

				while (wIndex < unemptyCells[spinId * (size + 1)]) {
					meanFieldMembers[wIndex + blockId * size] = meanFieldMember(
							mat, spins + blockId * size, spinId,
							unemptyCells[spinId * (size + 1) + wIndex + 1],
							size);
					wIndex = wIndex + blockDim.x;
				}
				__syncthreads();

				// Parallelized mean-field computation
				long long offset = 1;
				while (offset < unemptyCells[spinId * (size + 1)]) {
					wIndex = thrId;
					while ((wIndex * 2 + 1) * offset
							< unemptyCells[spinId * (size + 1)]) {
						meanFieldMembers[wIndex * 2 * offset + blockId * size] +=
								meanFieldMembers[(wIndex * 2 + 1) * offset
										+ blockId * size];
						wIndex = wIndex + blockDim.x;
					}
					offset *= 2;
					__syncthreads();
				}
				__syncthreads();

				// Mean-field calculation complete - write new spin and delta
				if (thrId == 0) {
					float meanField = meanFieldMembers[blockId * size];
					float old = spins[spinId + blockId * size];
					if (temp[blockId] > 0) {
						spins[spinId + blockId * size] = -1
								* tanh(meanField / temp[blockId]) * linearCoef
								+ spins[spinId + blockId * size]
										* (1 - linearCoef);
					} else if (meanField > 0)
						spins[spinId + blockId * size] = -1;
					else
						spins[spinId + blockId * size] = 1;

					if (proceedThreshold
							< fabs(old - spins[spinId + blockId * size]))
						proceedFlags[blockId] = true; // Too big delta. One more iteration needed
				}
				__syncthreads();
			}
		} while (proceedFlags[blockId]);
	} while (temp[blockId] >= 0);
}

void CudaAnnealing::anneal(float pStep, float linearCoef) {
	cudaKernelAnneal<<<blockCount, blockSize>>>(devMat, devSpins, size, devTemp,
			pStep, meanFieldMembers, continueIteration, minDiff, devUnemptyMat, linearCoef);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	checkError(err, "Kernel at cudaPull");
}
