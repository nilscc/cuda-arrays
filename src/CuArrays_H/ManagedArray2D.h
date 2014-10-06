#pragma once

#include "ManagedArrayND.h"

namespace CuArrays
{

template <typename T> class Array2D;

template <typename T>
class ManagedArray2D : public ManagedArrayND<T, Array2D>
{
public:

    ManagedArray2D(Array2D<T> & symbol)
        : ManagedArrayND<T, Array2D>(symbol)
    {
    }

    void mallocDevice(int N, int M)
    {
        Array2D<T> arr(N, M);
        ManagedArrayND<T, Array2D>::mallocDevice(arr, N, M);
    }

    void mallocHost  (int N, int M)
    {
        ManagedArrayND<T, Array2D>::mallocHost(N, M);
    }

    void malloc      (int N, int M)
    {
        mallocDevice(N, M);
        mallocHost(N, M);
    }

          T& operator()(unsigned int i, unsigned j)       { return ManagedArrayND<T, Array2D>::operator()(i, j); }
    const T& operator()(unsigned int i, unsigned j) const { return ManagedArrayND<T, Array2D>::operator()(i, j); }
};

}
