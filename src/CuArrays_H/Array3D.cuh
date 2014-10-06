#pragma once

#include "Array2D.cuh"

namespace CuArrays
{

template <typename T> class ManagedArray3D;
template <typename T, template <typename> class DeviceArray> class ManagedArrayND;

/*
 * 3D arrays
 *
 */

template <typename T>
class Array3D
{
    friend class ManagedArrayND<T, Array3D>;

    private:

        Array2D<T> array2d;

        int _N, _M, _O;

    protected:

        void setPitch(size_t);
        void setDPtr(T*);

    public:

        __device__ __host__ Array3D();
                   __host__ Array3D(int N, int M, int O);

        __device__ int N() const;
        __device__ int M() const;
        __device__ int O() const;

        __device__ T& get(unsigned int i, unsigned int j, unsigned int k);
        __device__ const T& get(unsigned int i, unsigned int j, unsigned int k) const;

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
    : array2d(Array2D<T>(N, M * O))
    , _N(N)
    , _M(M)
    , _O(O)
{
}

template <typename T>
__device__ __host__
Array3D<T>::Array3D()
    : array2d(Array2D<T>())
{
}

template <typename T>
__host__
void Array3D<T>::setPitch(size_t pitch)
{
    array2d.setPitch(pitch);
}

template <typename T>
__host__
void Array3D<T>::setDPtr(T* dptr)
{
    array2d.setDPtr(dptr);
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
T& Array3D<T>::get(unsigned int i, unsigned int j, unsigned int k)
{
    return array2d.get(i, _M * j + k);
}

template <typename T>
__device__
const T& Array3D<T>::get(unsigned int i, unsigned int j, unsigned int k) const
{
    return array2d.get(i, _M * j + k);
}

template <typename T>
__device__
T& Array3D<T>::operator()(unsigned int i, unsigned int j, unsigned int k)
{
    return get(i,j,k);
}

template <typename T>
__device__
const T& Array3D<T>::operator()(unsigned int i, unsigned int j, unsigned int k) const
{
    return get(i,j,k);
}

}
