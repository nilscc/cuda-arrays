#pragma once

#include "ManagedPitchedArray.h"

namespace CuArrays
{

template <typename T> class Array3D;

template <typename T>
class ManagedArray3D : public ManagedPitchedArray<T, Array3D>
{
private:

    int _M, _O;

public:

    ManagedArray3D(Array3D<T> & symbol)
        : ManagedPitchedArray<T, Array3D>(symbol)
    {
    }

    /*
     * Queries
     *
     */

    int N() const { return ManagedPitchedArray<T, Array3D>::_sizeX; }
    int M() const { return _M; }
    int O() const { return _O; }

    T& operator()(unsigned int i, unsigned int j, unsigned k)
    {
        return ManagedPitchedArray<T, Array3D>::get(i, j + _M * k);
    }

    const T& operator()(unsigned int i, unsigned int j, unsigned k) const
    {
        return ManagedPitchedArray<T, Array3D>::get()(i, j + _M * k);
    }

    inline T* devicePtrAt(size_t i, size_t j, size_t k)
    {
        return ManagedPitchedArray<T, Array3D>::devicePtrAt(i, j + _M * k);
    }

    inline const T* devicePtrAt(size_t i, size_t j, size_t k) const
    {
        return ManagedPitchedArray<T, Array3D>::devicePtrAt(i, j + _M * k);
    }

    /*
     * Malloc
     *
     */

    void mallocDevice(int N, int M, int O)
    {
        Array3D<T> arr(N, M, O);
        ManagedPitchedArray<T, Array3D>::mallocDevice(arr, N, M * O);
        _M = M;
        _O = O;
    }

    void mallocHost(int N, int M, int O)
    {
        ManagedPitchedArray<T, Array3D>::mallocHost(N, M * O);
        _M = M;
        _O = O;
    }

    void malloc(int N, int M, int O)
    {
        mallocDevice(N, M, O);
        mallocHost(N, M, O);
    }

};

}
