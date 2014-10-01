#pragma once

#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
#include "cuda_utils.h"

namespace CuArrays
{

/*
 * Class definition
 *
 */

template <typename T> class Array2D;

template <typename T>
class ManagedArray2D
{
    private:

        int _N, _M;

        Array2D<T> & symbol;

        T* d_ptr;
        size_t d_pitch;

        T* h_ptr;

    public:

        ManagedArray2D(Array2D<T> &symbol);

        int N() const;
        int M() const;

        T& get(unsigned int i, unsigned int j);
        const T& get(unsigned int i, unsigned int j) const;

        T& operator()(unsigned int i, unsigned int j);
        const T& operator()(unsigned int i, unsigned int j) const;

        void mallocDevice(int N, int M);
        void mallocHost(int N, int M);
        void malloc(int N, int M);

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
ManagedArray2D<T>::ManagedArray2D(Array2D<T> &symbol)
    : symbol(symbol)
{
    _N = 0;
    _M = 0;
    d_ptr = 0;
    d_pitch = 0;
    h_ptr = 0;
}

template <typename T>
int ManagedArray2D<T>::N() const { return _N; }
template <typename T>
int ManagedArray2D<T>::M() const { return _M; }

template <typename T>
T& ManagedArray2D<T>::get(unsigned int i, unsigned int j)
{
    return * (h_ptr + j * _N + i);
}
template <typename T>
const T& ManagedArray2D<T>::get(unsigned int i, unsigned int j) const
{
    return * (h_ptr + j * _N + i);
}

template <typename T>
T& ManagedArray2D<T>::operator()(unsigned int i, unsigned int j)
{
    return get(i,j);
}
template <typename T>
const T& ManagedArray2D<T>::operator()(unsigned int i, unsigned int j) const
{
    return get(i,j);
}

template <typename T>
void ManagedArray2D<T>::mallocDevice(int N, int M)
{
    using namespace cuda_utils;

    assert(d_ptr == 0);

    _N = N;
    _M = M;

    cudaVerify( cudaMallocPitch(&d_ptr, &d_pitch, N * sizeof(T), M) );

    Array2D<T> arr(d_ptr, d_pitch, N, M);

    cudaVerify( cudaMemcpyToSymbol(symbol, &arr, sizeof(Array2D<T>)) );
}

template <typename T>
void ManagedArray2D<T>::mallocHost(int N, int M)
{
    using namespace cuda_utils;

    assert(h_ptr == 0);

    _N = N;
    _M = M;

    cudaVerify( cudaMallocHost(&h_ptr, N * M * sizeof(T)) );
}

template <typename T>
void ManagedArray2D<T>::malloc(int N, int M)
{
    mallocDevice(N, M);
    mallocHost(N, M);
}

template <typename T>
void ManagedArray2D<T>::freeDevice()
{
    using namespace cuda_utils;

    assert(d_ptr != 0);

    Array2D<T> arr;

    cudaVerify( cudaMemcpyFromSymbol(&arr, symbol, sizeof(Array2D<T>)) );
    cudaVerify( cudaFree(arr.d_ptr) );

    d_ptr = 0;
}

template <typename T>
void ManagedArray2D<T>::freeHost()
{
    using namespace cuda_utils;

    assert(h_ptr != 0);

    cudaVerify( cudaFreeHost(h_ptr) );

    h_ptr = 0;
}

template <typename T>
void ManagedArray2D<T>::free()
{
    freeDevice();
    freeHost();
}

template <typename T>
void ManagedArray2D<T>::copyToDevice()
{
    using namespace cuda_utils;

    cudaVerify( cudaMemcpy2D(d_ptr, d_pitch,
                             h_ptr, _N * sizeof(T),
                             _N * sizeof(T), _M,
                             cudaMemcpyHostToDevice) );
}

template <typename T>
void ManagedArray2D<T>::copyToDeviceAsync()
{
    using namespace cuda_utils;

    cudaVerify( cudaMemcpy2DAsync(d_ptr, d_pitch,
                                  h_ptr, _N * sizeof(T),
                                  _N * sizeof(T), _M,
                                  cudaMemcpyHostToDevice) );
}

template <typename T>
void ManagedArray2D<T>::copyFromDevice()
{
    using namespace cuda_utils;

    cudaVerify( cudaMemcpy2D(h_ptr, _N * sizeof(T),
                             d_ptr, d_pitch,
                             _N * sizeof(T), _M,
                             cudaMemcpyDeviceToHost) );
}

template <typename T>
void ManagedArray2D<T>::copyFromDeviceAsync()
{
    using namespace cuda_utils;

    cudaVerify( cudaMemcpy2DAsync(h_ptr, _N * sizeof(T),
                                  d_ptr, d_pitch,
                                  _N * sizeof(T), _M,
                                  cudaMemcpyDeviceToHost) );
}

}
