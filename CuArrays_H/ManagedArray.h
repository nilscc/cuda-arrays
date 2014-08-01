#pragma once

#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
#include "cuda_utils.h"

template <typename T> class Array;

/*
 * Class definition
 *
 */

template <typename T>
class ManagedArray
{
    private:

        int _N;

        Array<T> & symbol;

        T* d_ptr;
        T* h_ptr;

    public:

        ManagedArray(Array<T> &symbol);

        T& get(unsigned int i);
        const T& get(unsigned int i) const;

        T& operator()(unsigned int i);
        const T& operator()(unsigned int i) const;

        int N() const;

        void mallocDevice(int N);
        void mallocHost(int N);
        void malloc(int N);

        void freeDevice();
        void freeHost();
        void free();

        void copyToDevice();
        void copyToDeviceAsync();

        void copyFromDevice();
        void copyFromDeviceAsync();
};

/*
 * Implementations
 *
 */

template <typename T>
ManagedArray<T>::ManagedArray(Array<T> &symbol)
    : symbol(symbol)
{
    _N = 0;
    d_ptr = 0;
    h_ptr = 0;
}

template <typename T>
T& ManagedArray<T>::get(unsigned int i)
{
    return * (h_ptr + i);
}
template <typename T>
const T& ManagedArray<T>::get(unsigned int i) const
{
    return * (h_ptr + i);
}

template <typename T>
T& ManagedArray<T>::operator()(unsigned int i)
{
    return get(i);
}
template <typename T>
const T& ManagedArray<T>::operator()(unsigned int i) const
{
    return get(i);
}

template <typename T>
int ManagedArray<T>::N() const { return _N; }

template <typename T>
void ManagedArray<T>::mallocDevice(int N)
{
    assert(d_ptr == 0);

    _N = N;

    cudaVerify( cudaMalloc(&d_ptr, _N * sizeof(T)) );

    Array<T> arr(d_ptr, _N);

    cudaVerify( cudaMemcpyToSymbol(symbol, &arr, sizeof(Array<T>)) );
}

template <typename T>
void ManagedArray<T>::mallocHost(int N)
{
    assert(h_ptr == 0);

    _N = N;

    cudaVerify( cudaMallocHost(&h_ptr, _N * sizeof(T)) );
}

template <typename T>
void ManagedArray<T>::malloc(int N)
{
    mallocDevice(N);
    mallocHost(N);
}

template <typename T>
void ManagedArray<T>::freeDevice()
{
    assert(d_ptr != 0);

    Array<T> arr;

    cudaVerify( cudaMemcpyFromSymbol(&arr, symbol, sizeof(Array<T>)) );
    cudaVerify( cudaFree(arr.d_ptr) );

    d_ptr = 0;
}

template <typename T>
void ManagedArray<T>::freeHost()
{
    assert(h_ptr != 0);

    cudaVerify( cudaFreeHost(h_ptr) );

    h_ptr = 0;
}

template <typename T>
void ManagedArray<T>::free()
{
    freeDevice();
    freeHost();
}

template <typename T>
void ManagedArray<T>::copyToDevice()
{
    cudaVerify( cudaMemcpy(d_ptr, h_ptr, _N * sizeof(T), cudaMemcpyHostToDevice) );
}

template <typename T>
void ManagedArray<T>::copyToDeviceAsync()
{
    cudaVerify( cudaMemcpyAsync(d_ptr, h_ptr, _N * sizeof(T), cudaMemcpyHostToDevice) );
}

template <typename T>
void ManagedArray<T>::copyFromDevice()
{
    cudaVerify( cudaMemcpy(h_ptr, d_ptr, _N * sizeof(T), cudaMemcpyDeviceToHost) );
}

template <typename T>
void ManagedArray<T>::copyFromDeviceAsync()
{
    cudaVerify( cudaMemcpyAsync(h_ptr, d_ptr, _N * sizeof(T), cudaMemcpyDeviceToHost) );
}
