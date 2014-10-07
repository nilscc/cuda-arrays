#include <CuArrays>
#include <cassert>

namespace Test
{
    __constant__ CuArrays::Array<int> test;
}

__global__ void device_test()
{
    using namespace Test;

    assert(test(0) == 1);
    assert(test(1) == 1);
}

void host_test(CuArrays::ManagedArray<int> &test)
{
    for (int i = 0; i < test.N(); i++)
        test(i) = 1;

    test.copyToDevice();

    assert(test(0) == 1);
    assert(test(1) == 1);
}

int main()
{
    CuArrays::ManagedArray<int> test(Test::test);
    test.malloc(2);

    host_test(test);
    device_test<<<1,1>>>();

    cudaDeviceSynchronize();
    assert(cudaGetLastError() == cudaSuccess);

    test.free();

    return 0;
}
