#pragma once

namespace CuArrays
{

template <typename T, template <typename> class DeviceArray> class ManagedArrayND;
template <typename T> class ManagedArray2D;

template <typename T> class ManagedArray3D;
template <typename T> class Array3D;

/*
 * Class definition
 *
 */

template <typename T>
class Array2D
{
    friend class ManagedArrayND<T, Array2D>;

    friend class Array3D<T>;
    friend class ManagedArray3D<T>;

    private:

        char* d_ptr;
        size_t d_pitch;

        void setPitch(size_t);
        void setDPtr(T*);

        int _N, _M;

    public:

        __host__ __device__ Array2D();
        __host__            Array2D(int N, int M);

        __device__ int N() const;
        __device__ int M() const;

        __device__ T& operator()(unsigned int i, unsigned int j);
        __device__ const T& operator()(unsigned int i, unsigned int j) const;

        __device__ T& get(unsigned int i, unsigned int j);
        __device__ const T& get(unsigned int i, unsigned int j) const;
};

/*
 * Implementations
 *
 */

template <typename T>
__host__
Array2D<T>::Array2D(int N, int M)
    : d_ptr(0)
    , d_pitch(0)
    , _N(N)
, _M(M)
{
}

template <typename T>
__host__ __device__
Array2D<T>::Array2D()
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

// setter
template <typename T>
__host__
void Array2D<T>::setPitch(size_t pitch)
{
    d_pitch = pitch;
}

// setter
template <typename T>
__host__
void Array2D<T>::setDPtr(T* dptr)
{
    d_ptr = (char*) dptr;
}

template <typename T>
__device__
T& Array2D<T>::operator()(unsigned int i, unsigned int j)
{
    return * ((T*) ((char*) d_ptr + j * d_pitch) + i);
}

template <typename T>
__device__
const T& Array2D<T>::operator()(unsigned int i, unsigned int j) const
{
    return * ((T*) ((char*) d_ptr + j * d_pitch) + i);
}

template <typename T>
__device__
T& Array2D<T>::get(unsigned int i, unsigned int j)
{
    return operator()(i,j);
}

template <typename T>
__device__ const T& Array2D<T>::get(unsigned int i, unsigned int j) const
{
    return operator()(i,j);
}

}
