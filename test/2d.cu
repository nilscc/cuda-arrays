#include <CuArrays>
#include <cassert>

namespace Test
{
    __constant__ CuArrays::Array2D<int> test;
}

__global__ void device_test()
{
    using namespace Test;

    assert(test(0,0) == 1);
    assert(test(1,1) == 1);

    printf("OK!\n");
}

void host_test(CuArrays::ManagedArray2D<int> &test)
{
    assert(test(0,0) == 1);
    assert(test(1,1) == 1);
}

int main()
{
    CuArrays::ManagedArray2D<int> test(Test::test);
    test.malloc(2, 2);

    for (int i = 0; i < test.N(); i++)
        for (int j = 0; j < test.M(); j++)
            test(i,j) = 1;

    test.copyToDevice();

    // validate data
    host_test(test);
    device_test<<<1,1>>>();

    cudaDeviceSynchronize();

    test.free();

    assert(cudaGetLastError() == cudaSuccess);

    return 0;
}
