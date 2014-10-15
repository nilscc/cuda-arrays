#pragma once

#include "ManagedPitchedArray.h"

namespace CuArrays
{

template <typename T> class Array2D;

template <typename T>
class ManagedArray2D : public ManagedPitchedArray<T, Array2D>
{
public:

    ManagedArray2D(Array2D<T> & symbol)
        : ManagedPitchedArray<T, Array2D>(symbol)
    {
    }

    void mallocDevice(int N, int M)
    {
        Array2D<T> arr(N, M);
        ManagedPitchedArray<T, Array2D>::mallocDevice(arr, N, M);
    }

    void mallocHost  (int N, int M)
    {
        ManagedPitchedArray<T, Array2D>::mallocHost(N, M);
    }

    void malloc      (int N, int M)
    {
        mallocDevice(N, M);
        mallocHost(N, M);
    }

    int N() const { return ManagedPitchedArray<T,Array2D>::_sizeX; }
    int M() const { return ManagedPitchedArray<T,Array2D>::_sizeY; }

    inline       T* devicePtrAt(int i, int j)       { return ManagedPitchedArray<T, Array2D>::devicePtrAt(i,j); }
    inline const T* devicePtrAt(int i, int j) const { return ManagedPitchedArray<T, Array2D>::devicePtrAt(i,j); }

          T& operator()(unsigned int i, unsigned j)       { return ManagedPitchedArray<T, Array2D>::get(i, j); }
    const T& operator()(unsigned int i, unsigned j) const { return ManagedPitchedArray<T, Array2D>::get(i, j); }
};

}
