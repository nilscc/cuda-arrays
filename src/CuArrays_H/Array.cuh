#pragma once

namespace CuArrays
{

template <typename T> class ManagedArray;

/*
 * Class definition
 *
 */

template <typename T>
class Array
{
    friend class ManagedArray<T>;

    private:

        T* d_ptr;

        int _N;

        __host__ Array(T* d_ptr, int N);

    public:

        __host__ __device__ Array();

        __device__ T& get(unsigned int i);

        __device__ T& operator()(unsigned int i);

        __device__ int N() const;
};

/*
 * Implementations
 *
 */


template <typename T>
__host__
Array<T>::Array(T* d_ptr, int N)
    : d_ptr(d_ptr)
    , _N(N)
{
}

template <typename T>
__host__ __device__
Array<T>::Array()
{
}

template <typename T>
__device__
T& Array<T>::get(unsigned int i)
{
    return * (d_ptr + i);
}

template <typename T>
__device__
T& Array<T>::operator()(unsigned int i)
{
    return get(i);
}

}
