#pragma once

namespace CuArrays
{

template <typename T>
class PitchedArray
{
protected:

    size_t d_pitch;
    char *d_ptr;

public:

    void setPitch(size_t p) { d_pitch = p; }
    void setDPtr(T* ptr) { d_ptr = (char*) ptr; }

    __device__ T& get(unsigned int i, unsigned int j)
    {
        return * ((T*) ((char*) d_ptr + j * d_pitch) + i);
    }

    __device__ const T& get(unsigned int i, unsigned int j) const
    {
        return * ((T*) ((char*) d_ptr + j * d_pitch) + i);
    }
};

}
