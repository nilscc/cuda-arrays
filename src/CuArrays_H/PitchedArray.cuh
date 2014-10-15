#pragma once

namespace CuArrays
{

template <typename T>
class PitchedArray
{
protected:

    size_t d_pitch;
    T *d_ptr;

public:

    void setPitch(size_t p) { d_pitch = p; }
    void setDPtr(T* ptr)    { d_ptr   = ptr; }

    __device__ __host__
    static T* devicePtrAt(T *d_ptr, size_t d_pitch, unsigned int i, unsigned int j)
    {
        return (T*) ((char*) d_ptr + j * d_pitch) + i;
    }

    __device__ __host__
    inline T* devicePtrAt(unsigned int i, unsigned int j)
    {
        return PitchedArray<T>::devicePtrAt(d_ptr, d_pitch, i, j);
    }

    __device__ __host__
    inline const T* devicePtrAt(unsigned int i, unsigned int j) const
    {
        return PitchedArray<T>::devicePtrAt(d_ptr, d_pitch, i, j);
    }

    __device__
    T& get(unsigned int i, unsigned int j)
    {
        return *devicePtrAt(i,j);
    }

    __device__
    const T& get(unsigned int i, unsigned int j) const
    {
        return *devicePtrAt(i,j);
    }
};

}
