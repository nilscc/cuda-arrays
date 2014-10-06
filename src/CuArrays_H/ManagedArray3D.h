#pragma once

#include "ManagedArrayND.h"

namespace CuArrays
{

template <typename T> class Array3D;

template <typename T>
class ManagedArray3D : public ManagedArrayND<T, Array3D>
{
private:

    int _M, _O;

public:

    ManagedArray3D(Array3D<T> & symbol)
        : ManagedArrayND<T, Array3D>(symbol)
    {
    }

    /*
     * Queries
     *
     */

    int N() const { return ManagedArrayND<T, Array3D>::N(); }
    int M() const { return _M; }
    int O() const { return _O; }

    T& get(unsigned int i, unsigned int j, unsigned k)
    {
        return ManagedArrayND<T, Array3D>::get(i + _M * j, k);
    }

    const T& get(unsigned int i, unsigned int j, unsigned k) const
    {
        return ManagedArrayND<T, Array3D>::get(i + _M * j, k);
    }

    T& operator()(unsigned int i, unsigned int j, unsigned k)
    {
        return ManagedArrayND<T, Array3D>::operator()(i + _M * j, k);
    }

    const T& operator()(unsigned int i, unsigned int j, unsigned k) const
    {
        return ManagedArrayND<T, Array3D>::operator()(i + _M * j, k);
    }

    /*
     * Malloc
     *
     */

    void mallocDevice(int N, int M, int O)
    {
        Array3D<T> arr(N, M, O);
        ManagedArrayND<T, Array3D>::mallocDevice(arr, N, M * O);
        _M = M;
        _O = O;
    }

    void mallocHost(int N, int M, int O)
    {
        ManagedArrayND<T, Array3D>::mallocHost(N, M * O);
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
