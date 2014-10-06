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

template <typename T, template <typename> class DeviceArray>
class ManagedArrayND
{
public:

    ManagedArrayND(DeviceArray<T> &symbol);

    void freeDevice();
    void freeHost();
    void free();

    void copyToDevice();
    void copyToDeviceAsync();

    void copyFromDevice();
    void copyFromDeviceAsync();

    int _N, _M;

    DeviceArray<T> & symbol;

private:

    T* d_ptr;
    size_t d_pitch;

    T* h_ptr;

protected:

    int N() const;
    int M() const;

    T& get(unsigned int i, unsigned int j);
    const T& get(unsigned int i, unsigned int j) const;

    T& operator()(unsigned int i, unsigned int j);
    const T& operator()(unsigned int i, unsigned int j) const;

    void mallocDevice(DeviceArray<T> &arr, int N, int M);
    void mallocHost(int N, int M);
    void malloc(int N, int M);

};

/*
 * Implementations
 *
 */

template <typename T, template <typename> class DeviceArray>
ManagedArrayND<T, DeviceArray>::ManagedArrayND(DeviceArray<T> &symbol)
    : symbol(symbol)
{
    _N = 0;
    _M = 0;
    d_ptr = 0;
    d_pitch = 0;
    h_ptr = 0;
}

template <typename T, template <typename> class DeviceArray>
int ManagedArrayND<T, DeviceArray>::N() const { return _N; }
template <typename T, template <typename> class DeviceArray>
int ManagedArrayND<T, DeviceArray>::M() const { return _M; }

template <typename T, template <typename> class DeviceArray>
T& ManagedArrayND<T, DeviceArray>::get(unsigned int i, unsigned int j)
{
    return * (h_ptr + j * _N + i);
}
template <typename T, template <typename> class DeviceArray>
const T& ManagedArrayND<T, DeviceArray>::get(unsigned int i, unsigned int j) const
{
    return * (h_ptr + j * _N + i);
}

template <typename T, template <typename> class DeviceArray>
T& ManagedArrayND<T, DeviceArray>::operator()(unsigned int i, unsigned int j)
{
    return get(i,j);
}
template <typename T, template <typename> class DeviceArray>
const T& ManagedArrayND<T, DeviceArray>::operator()(unsigned int i, unsigned int j) const
{
    return get(i,j);
}

template <typename T, template <typename> class DeviceArray>
void ManagedArrayND<T, DeviceArray>::mallocDevice(DeviceArray<T> &arr, int N, int M)
{
    using namespace cuda_utils;

    assert(d_ptr == 0);

    _N = N;
    _M = M;

    cudaVerify( cudaMallocPitch(&d_ptr, &d_pitch, N * sizeof(T), M) );

    arr.setPitch(d_pitch);
    arr.setDPtr(d_ptr);

    cudaVerify( cudaMemcpyToSymbol(symbol, &arr, sizeof(DeviceArray<T>)) );
}

template <typename T, template <typename> class DeviceArray>
void ManagedArrayND<T, DeviceArray>::mallocHost(int N, int M)
{
    using namespace cuda_utils;

    assert(h_ptr == 0);

    _N = N;
    _M = M;

    cudaVerify( cudaMallocHost(&h_ptr, N * M * sizeof(T)) );
}

template <typename T, template <typename> class DeviceArray>
void ManagedArrayND<T, DeviceArray>::freeDevice()
{
    using namespace cuda_utils;

    assert(d_ptr != 0);

    DeviceArray<T> arr;

    cudaVerify( cudaMemcpyFromSymbol(&arr, symbol, sizeof(DeviceArray<T>)) );
    cudaVerify( cudaFree(arr.d_ptr) );

    d_ptr = 0;
}

template <typename T, template <typename> class DeviceArray>
void ManagedArrayND<T, DeviceArray>::freeHost()
{
    using namespace cuda_utils;

    assert(h_ptr != 0);

    cudaVerify( cudaFreeHost(h_ptr) );

    h_ptr = 0;
}

template <typename T, template <typename> class DeviceArray>
void ManagedArrayND<T, DeviceArray>::free()
{
    freeDevice();
    freeHost();
}

template <typename T, template <typename> class DeviceArray>
void ManagedArrayND<T, DeviceArray>::copyToDevice()
{
    using namespace cuda_utils;

    cudaVerify( cudaMemcpy2D(d_ptr, d_pitch,
                             h_ptr, _N * sizeof(T),
                             _N * sizeof(T), _M,
                             cudaMemcpyHostToDevice) );
}

template <typename T, template <typename> class DeviceArray>
void ManagedArrayND<T, DeviceArray>::copyToDeviceAsync()
{
    using namespace cuda_utils;

    cudaVerify( cudaMemcpy2DAsync(d_ptr, d_pitch,
                                  h_ptr, _N * sizeof(T),
                                  _N * sizeof(T), _M,
                                  cudaMemcpyHostToDevice) );
}

template <typename T, template <typename> class DeviceArray>
void ManagedArrayND<T, DeviceArray>::copyFromDevice()
{
    using namespace cuda_utils;

    cudaVerify( cudaMemcpy2D(h_ptr, _N * sizeof(T),
                             d_ptr, d_pitch,
                             _N * sizeof(T), _M,
                             cudaMemcpyDeviceToHost) );
}

template <typename T, template <typename> class DeviceArray>
void ManagedArrayND<T, DeviceArray>::copyFromDeviceAsync()
{
    using namespace cuda_utils;

    cudaVerify( cudaMemcpy2DAsync(h_ptr, _N * sizeof(T),
                                  d_ptr, d_pitch,
                                  _N * sizeof(T), _M,
                                  cudaMemcpyDeviceToHost) );
}

}
