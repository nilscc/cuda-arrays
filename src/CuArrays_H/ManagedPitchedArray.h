#pragma once

#include "PitchedArray.cuh"

namespace CuArrays
{

template <typename T, template <typename> class DeviceArray>
class ManagedPitchedArray
{
protected:

    size_t _sizeX, _sizeY, _dpitch;

    DeviceArray<T> &_symbol;
    T *_hptr, *_dptr;

public:

    ManagedPitchedArray(DeviceArray<T> &symbol);

private: // disable copy constructor
    ManagedPitchedArray(const ManagedPitchedArray &copy); // no implementation

protected:

    void mallocDevice(DeviceArray<T> &arr, int N, int M);
    void mallocHost(int N, int M);
    void malloc(int N, int M);

public:

    void free();

    void copyToDevice();
    void copyToDeviceAsync();
    void copyFromDevice();
    void copyFromDeviceAsync();

protected:

          T& get(size_t i, size_t j);
    const T& get(size_t i, size_t j) const;

    inline       T* devicePtrAt(size_t i, size_t j);
    inline const T* devicePtrAt(size_t i, size_t j) const;
};

/*
 * Implementations
 *
 */

template <typename T, template <typename> class DeviceArray>
ManagedPitchedArray<T, DeviceArray>::ManagedPitchedArray(
        DeviceArray<T> &symbol)
    : _symbol(symbol)
    , _hptr(0)
    , _dptr(0)
    , _sizeX(0)
    , _sizeY(0)
{
    // make sure DeviceArray has PitchedArray as base class
    (void) static_cast<PitchedArray<T>*>((DeviceArray<T>*)0);
}

template <typename T, template <typename> class DeviceArray>
void ManagedPitchedArray<T, DeviceArray>::mallocDevice(
        DeviceArray<T> &arr, int N, int M)
{
    using namespace cuda_utils;

    // make sure we haven't initialized device yet
    assert(_dptr == 0);

    // make sure dimensions match with host
    assert(_hptr == 0 || (N == _sizeX && M == _sizeY));
    _sizeX = N;
    _sizeY = M;

    cudaVerify( cudaMallocPitch(&_dptr, &_dpitch, N * sizeof(T), M) );

    arr.setPitch(_dpitch);
    arr.setDPtr(_dptr);

    cudaVerify( cudaMemcpyToSymbol(_symbol, &arr, sizeof(DeviceArray<T>)) );
}

template <typename T, template <typename> class DeviceArray>
void ManagedPitchedArray<T, DeviceArray>::mallocHost(int N, int M)
{
    using namespace cuda_utils;

    // make sure we haven't initialized host yet
    assert(_hptr == 0);

    // make sure dimensions match device pointer
    assert(_dptr == 0 || (N == _sizeX && M == _sizeY));
    _sizeX = N;
    _sizeY = M;

    cudaVerify( cudaMallocHost(&_hptr, _sizeX * _sizeY * sizeof(T)) );
}

template <typename T, template <typename> class DeviceArray>
void ManagedPitchedArray<T, DeviceArray>::free()
{
    using namespace cuda_utils;

    if (_hptr != 0)
        cudaVerify( cudaFreeHost(_hptr) );

    if (_dptr != 0)
    {
        // store empty array at symbol
        DeviceArray<T> arr;
        cudaVerify( cudaMemcpyToSymbol(_symbol, &arr, sizeof(DeviceArray<T>)) );

        // free actual memory
        cudaVerify( cudaFree(_dptr) );
    }

    _hptr = 0;
    _dptr = 0;
}

template <typename T, template <typename> class DeviceArray>
void ManagedPitchedArray<T, DeviceArray>::copyToDevice()
{
    using namespace cuda_utils;

    cudaVerify( cudaMemcpy2D(
                _dptr, _dpitch,                 // dest + pitch
                _hptr, _sizeX * sizeof(T),      // src  + pitch
                _sizeX * sizeof(T),             // X size
                _sizeY,                         // Y size
                cudaMemcpyHostToDevice) );
}

template <typename T, template <typename> class DeviceArray>
void ManagedPitchedArray<T, DeviceArray>::copyToDeviceAsync()
{
    using namespace cuda_utils;

    cudaVerify( cudaMemcpy2DAsync(
                _dptr, _dpitch,                 // dest + pitch
                _hptr, _sizeX * sizeof(T),      // src  + pitch
                _sizeX * sizeof(T),             // X size
                _sizeY,                         // Y size
                cudaMemcpyHostToDevice) );
}



template <typename T, template <typename> class DeviceArray>
void ManagedPitchedArray<T, DeviceArray>::copyFromDevice()
{
    using namespace cuda_utils;

    cudaVerify( cudaMemcpy2DAsync(
                _hptr, _sizeX * sizeof(T),      // dest + pitch
                _dptr, _dpitch,                 // src  + pitch
                _sizeX * sizeof(T),             // X size
                _sizeY,                         // Y size
                cudaMemcpyDeviceToHost) );
}


template <typename T, template <typename> class DeviceArray>
void ManagedPitchedArray<T, DeviceArray>::copyFromDeviceAsync()
{
    using namespace cuda_utils;

    cudaVerify( cudaMemcpy2DAsync(
                _hptr, _sizeX * sizeof(T),      // dest + pitch
                _dptr, _dpitch,                 // src  + pitch
                _sizeX * sizeof(T),             // X size
                _sizeY,                         // Y size
                cudaMemcpyDeviceToHost) );
}

template <typename T, template <typename> class DeviceArray>
T& ManagedPitchedArray<T, DeviceArray>::get(size_t i, size_t j)
{
    return *(_hptr + j * _sizeX + i);
}

template <typename T, template <typename> class DeviceArray>
const T& ManagedPitchedArray<T, DeviceArray>::get(size_t i, size_t j) const
{
    return *(_hptr + j * _sizeX + i);
}

template <typename T, template <typename> class DeviceArray>
T* ManagedPitchedArray<T, DeviceArray>::devicePtrAt(size_t i, size_t j)
{
    return DeviceArray<T>::devicePtrAt(_dptr, _dpitch, i, j);
}

template <typename T, template <typename> class DeviceArray>
const T* ManagedPitchedArray<T, DeviceArray>::devicePtrAt(size_t i, size_t j) const
{
    return DeviceArray<T>::devicePtrAt(_dptr, _dpitch, i, j);
}

}
