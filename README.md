# CuArrays: Multidimensional cuda arrays with proper memory management

This is a headers only library. To use it, simple include `CuArrays` in your
.cu or .cpp files. See the example below.

To compile, add the `cuda-arrays` folder to your include path, e.g.:

    nvcc -I$HOME/local/include/cuda-arrays example.cu

For other compilers don't forget to include the CUDA runtime headers:

    g++ -I$HOME/local/include/cuda-arrays \
        -I/usr/local/cuda/include example.cpp


# Example:

    #include <CuArrays>
    
    namespace Device
    {
        // Arrays only contain the pointers to global device memory, so they
        // can be stored in constant device memory for quick access:
        __constant__ Array<double>   test1d;
        __constant__ Array2D<double> test2d;
    }
    
    __global__ void test_kernel()
    {
        using namespace Device;
    
        int i = threadIdx.x;
        int j = threadIdx.y;
    
        if (j == 0)
            test1d(i) *= 2;
    
        test2d(i,j) *= 5;
    }
    
    int main()
    {
        // link device and host arrays
        ManagedArray<double>   test1d(Device::test1d);
        ManagedArray2D<double> test2d(Device::test2d);
    
        // allocate memory (both on host and device)
        test1d.malloc(10);
        test2d.malloc(10,10);
    
        // fill host with data
        for (int i = 0; i < 10; i++)
        {
            test1d(i) = i;
            for (int j = 0; j < 10; j++)
            {
                test2d(i, j) = i + j;
            }
        }
    
        // copy to device
        test1d.copyToDeviceAsync();
        test2d.copyToDeviceAsync();
    
        // run test kernel
        test_kernel<<<1, dim3(10,10) >>>();
    
        // copy data back
        test1d.copyFromDeviceAsync();
        test2d.copyFromDeviceAsync();
    
        // output 1D array
        printf("test1d = ");
        for (int i = 0; i < 10; i++)
            printf("%.2e ", test1d(i));
        printf("\n\n");
    
        // output 2D array
        for (int i = 0; i < 10; i++)
        {
            printf("test2d[%i] = ", i);
            for (int j = 0; j < 10; j++)
            {
                printf("%.2e ", test2d(i, j));
            }
            printf("\n");
        }
    
        // free all memory
        test1d.free();
        test2d.free();
    
        return 0;
    }
