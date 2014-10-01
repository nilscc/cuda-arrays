#pragma once

#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
#include "cuda_utils.h"

namespace CuArrays
{

template <typename T> class Array3D;

template <typename T>
class ManagedArray3D
{
    private:

        Array3D<T> & symbol;

        int _N, _M, _O;

        T* h_ptr;

    public:

        ManagedArray3D(Array3D<T> & symbol);

        T& get(unsigned int i, unsigned int j, unsigned int k);
        const T& get(unsigned int i, unsigned int j, unsigned int k) const;

        T& operator()(unsigned int i, unsigned int j, unsigned int k);
        const T& operator()(unsigned int i, unsigned int j, unsigned int k) const;

        void mallocDevice(int N, int M, int O);
        void mallocHost(int N, int M, int O);
        void malloc(int N, int M, int O);

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
ManagedArray3D<T>::ManagedArray3D(Array3D<T> & symbol)
    : symbol(symbol)
{
    _N = 0;
    _M = 0;
    _O = 0;
    h_ptr = 0;
}

template <typename T>
T& ManagedArray3D<T>::get(unsigned int i, unsigned int j, unsigned int k)
{
    return * (h_ptr + i + _M * j + _M * _O * k);
}

template <typename T>
const T& ManagedArray3D<T>::get(unsigned int i, unsigned int j, unsigned int k) const
{
    return * (h_ptr + i + _M * j + _M * _O * k);
}

template <typename T>
T& ManagedArray3D<T>::operator()(unsigned int i, unsigned int j, unsigned int k)
{
    return get(i,j,k);
}

template <typename T>
const T& ManagedArray3D<T>::operator()(unsigned int i, unsigned int j, unsigned int k) const
{
    return get(i,j,k);
}

template <typename T>
void ManagedArray3D<T>::mallocDevice(int N, int M, int O)
{
    using namespace cuda_utils;

    _N = N;
    _M = M;
    _O = O;

    void *d_ptr;
    size_t d_pitch;

    cudaVerify( cudaMallocPitch(&d_ptr, &d_pitch, N * sizeof(T), M * O) );

    Array2D<T> arr2d(d_ptr, d_pitch, _N, _M * _O);
    Array3D<T> arr3d(arr2d, _N, _M, _O);

    cudaVerify( cudaMemcpyToSymbol(symbol, &arr3d, sizeof(Array3D<T>)) );
}

template <typename T>
void ManagedArray3D<T>::mallocHost(int N, int M, int O)
{
    using namespace cuda_utils;

    assert(h_ptr == 0);

    _N = N;
    _M = M;
    _O = O;

    assert(_N > 0);
    assert(_M > 0);
    assert(_O > 0);

    cudaVerify( cudaMallocHost(&h_ptr, _N * sizeof(T) * _M * _O) );
}

template <typename T>
void ManagedArray3D<T>::malloc(int N, int M, int O)
{
    mallocDevice(N, M, O);
    mallocHost(N, M, O);
}

template <typename T>
void ManagedArray3D<T>::freeDevice()
{
    using namespace cuda_utils;

    Array3D<T> arr;

    cudaVerify( cudaMemcpyFromSymbol(&arr, symbol, sizeof(Array3D<T>)) );
    cudaVerify( cudaFree(arr.array2d.d_ptr) );
}

template <typename T>
void ManagedArray3D<T>::freeHost()
{
    using namespace cuda_utils;

    assert(h_ptr != 0);

    cudaVerify( cudaFreeHost(h_ptr) );

    h_ptr = 0;
}

template <typename T>
void ManagedArray3D<T>::free()
{
    freeDevice();
    freeHost();
}

/*
template <typename T>
cudaPitchedPtr ManagedArray3D<T>::make_h_pitched_ptr()
{
    using namespace cuda_utils;

    cudaPitchedPtr p = make_cudaPitchedPtr(
            h_ptr,
            _N * sizeof(T),
            _M,
            _O);

    return p;
}

template <typename T>
cudaExtent ManagedArray3D<T>::make_extent()
{
    return make_cudaExtent(_N * sizeof(T), _M, _O);
}

template <typename T>
void ManagedArray3D<T>::copy(cudaMemcpyKind kind, cudaPitchedPtr src, cudaPitchedPtr dst)
{
    using namespace cuda_utils;

    cudaMemcpy3DParms par = make_3Dparms(kind, src, dst);
    cudaVerify( cudaMemcpy3D(&par) );
}

template <typename T>
void ManagedArray3D<T>::copyAsync(cudaMemcpyKind kind, cudaPitchedPtr src, cudaPitchedPtr dst)
{
    using namespace cuda_utils;

    cudaMemcpy3DParms par = make_3Dparms(kind, src, dst);
    cudaVerify( cudaMemcpy3DAsync(&par) );
}
*/

template <typename T>
void ManagedArray3D<T>::copyToDevice()
{
    //copy(cudaMemcpyHostToDevice, make_h_pitched_ptr(), d_pitched_ptr);
}

template <typename T>
void ManagedArray3D<T>::copyToDeviceAsync()
{
    //copyAsync(cudaMemcpyHostToDevice, make_h_pitched_ptr(), d_pitched_ptr);
}

template <typename T>
void ManagedArray3D<T>::copyFromDevice()
{
    //copy(cudaMemcpyDeviceToHost, d_pitched_ptr, make_h_pitched_ptr());
}

template <typename T>
void ManagedArray3D<T>::copyFromDeviceAsync()
{
    //copyAsync(cudaMemcpyDeviceToHost, d_pitched_ptr, make_h_pitched_ptr());
}

}
