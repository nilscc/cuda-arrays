#pragma once

#include "PitchedArray.cuh"

namespace CuArrays
{

/*
 * 3D arrays
 *
 */

template <typename T>
class Array3D : public PitchedArray<T>
{
    private:

        int _N, _M, _O;

    public:

        __device__ __host__ Array3D() {};
                   __host__ Array3D(int N, int M, int O);

        __device__ int N() const;
        __device__ int M() const;
        __device__ int O() const;

        __device__ T& operator()(unsigned int i, unsigned int j, unsigned int k);
        __device__ const T& operator()(unsigned int i, unsigned int j, unsigned int k) const;
};

/*
 * Implementations
 *
 */


template <typename T>
__host__
Array3D<T>::Array3D(int N, int M, int O)
    : _N(N)
    , _M(M)
    , _O(O)
{
}

template <typename T>
__device__
int Array3D<T>::N() const
{
    return _N;
}
template <typename T>
__device__
int Array3D<T>::M() const
{
    return _M;
}
template <typename T>
__device__
int Array3D<T>::O() const
{
    return _O;
}

template <typename T>
__device__
T& Array3D<T>::operator()(unsigned int i, unsigned int j, unsigned int k)
{
    return PitchedArray<T>::get(i, j + _M * k);
}

template <typename T>
__device__
const T& Array3D<T>::operator()(unsigned int i, unsigned int j, unsigned int k) const
{
    return PitchedArray<T>::get(i, j + _M * k);
}

}
