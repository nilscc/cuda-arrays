#pragma once

#include "PitchedArray.cuh"

namespace CuArrays
{

/*
 * Class definition
 *
 */

template <typename T>
class Array2D : public PitchedArray<T>
{
    private:

        int _N, _M;

    public:

        __host__ __device__ Array2D() {};
        __host__            Array2D(int N, int M);

        __device__ int N() const;
        __device__ int M() const;

        __device__ T& operator()(unsigned int i, unsigned int j);
        __device__ const T& operator()(unsigned int i, unsigned int j) const;
};

/*
 * Implementations
 *
 */

template <typename T>
__host__
Array2D<T>::Array2D(int N, int M)
    : _N(N)
    , _M(M)
{
}

// getter
template <typename T>
__device__
int Array2D<T>::N() const
{
    return _N;
}

// getter
template <typename T>
__device__
int Array2D<T>::M() const
{
    return _M;
}

template <typename T>
__device__
T& Array2D<T>::operator()(unsigned int i, unsigned int j)
{
    return PitchedArray<T>::get(i,j);
}

template <typename T>
__device__
const T& Array2D<T>::operator()(unsigned int i, unsigned int j) const
{
    return PitchedArray<T>::get(i,j);
}

}
