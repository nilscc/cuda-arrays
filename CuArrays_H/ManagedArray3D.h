#pragma once

#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
#include "cuda_utils.h"

template <typename T> class Array3D;

template <typename T>
class ManagedArray3D
{
    private:

        Array3D<T> & symbol;

        cudaPitchedPtr d_pitched_ptr;

        T* h_ptr;

        int _N, _M, _O;

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

    private:

        cudaPitchedPtr make_h_pitched_ptr();
        cudaExtent make_extent();
        cudaMemcpy3DParms make_3Dparms(
                cudaMemcpyKind kind, cudaPitchedPtr src, cudaPitchedPtr dst);

        void copy(cudaMemcpyKind kind, cudaPitchedPtr src, cudaPitchedPtr dst);
        void copyAsync(
                cudaMemcpyKind kind, cudaPitchedPtr src, cudaPitchedPtr dst);

    public:

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
    d_pitched_ptr.ptr = 0;
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
    assert(d_pitched_ptr.ptr == 0);

    _N = N;
    _M = M;
    _O = O;

    assert(_N > 0);
    assert(_M > 0);
    assert(_O > 0);

    cudaExtent extent = make_cudaExtent(_N * sizeof(T), _M, _O);

    cudaVerify( cudaMalloc3D(&d_pitched_ptr, extent) );

    Array3D<T> arr(d_pitched_ptr.ptr, d_pitched_ptr.pitch, _N, _M, _O);

    cudaVerify( cudaMemcpyToSymbol(symbol, &arr, sizeof(Array3D<T>)) );
}

template <typename T>
void ManagedArray3D<T>::mallocHost(int N, int M, int O)
{
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
    assert(d_pitched_ptr.ptr != 0);

    Array3D<T> arr;

    cudaVerify( cudaMemcpyFromSymbol(&arr, symbol, sizeof(Array3D<T>)) );
    cudaVerify( cudaFree(arr.d_ptr) );

    d_pitched_ptr.ptr = 0;
}

template <typename T>
void ManagedArray3D<T>::freeHost()
{
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

template <typename T>
cudaPitchedPtr ManagedArray3D<T>::make_h_pitched_ptr()
{
    cudaPitchedPtr p = make_cudaPitchedPtr(
            h_ptr,
            _N * sizeof(T),
            _M,
            _O);

    // p.ptr = (void*) h_ptr;
    // p.pitch = _N * sizeof(T);
    // p.xsize = _M;
    // p.ysize = _O;

    return p;
}

template <typename T>
cudaExtent ManagedArray3D<T>::make_extent()
{
    return make_cudaExtent(_N * sizeof(T), _M, _O);
}

template <typename T>
cudaMemcpy3DParms ManagedArray3D<T>::make_3Dparms(cudaMemcpyKind kind, cudaPitchedPtr src, cudaPitchedPtr dst)
{
    cudaMemcpy3DParms par = {0};

    par.srcPtr = src;
    par.dstPtr = dst;
    par.extent = make_extent();
    par.kind   = kind;

    return par;
}

template <typename T>
void ManagedArray3D<T>::copy(cudaMemcpyKind kind, cudaPitchedPtr src, cudaPitchedPtr dst)
{
    cudaMemcpy3DParms par = make_3Dparms(kind, src, dst);
    cudaVerify( cudaMemcpy3D(&par) );
}

template <typename T>
void ManagedArray3D<T>::copyAsync(cudaMemcpyKind kind, cudaPitchedPtr src, cudaPitchedPtr dst)
{
    cudaMemcpy3DParms par = make_3Dparms(kind, src, dst);
    cudaVerify( cudaMemcpy3DAsync(&par) );
}

template <typename T>
void ManagedArray3D<T>::copyToDevice()
{
    copy(cudaMemcpyHostToDevice, make_h_pitched_ptr(), d_pitched_ptr);
}

template <typename T>
void ManagedArray3D<T>::copyToDeviceAsync()
{
    copyAsync(cudaMemcpyHostToDevice, make_h_pitched_ptr(), d_pitched_ptr);
}

template <typename T>
void ManagedArray3D<T>::copyFromDevice()
{
    copy(cudaMemcpyDeviceToHost, d_pitched_ptr, make_h_pitched_ptr());
}

template <typename T>
void ManagedArray3D<T>::copyFromDeviceAsync()
{
    copyAsync(cudaMemcpyDeviceToHost, d_pitched_ptr, make_h_pitched_ptr());
}
