#include <CuArrays>
#include <cassert>

namespace Test
{
    __constant__ CuArrays::Array3D<int> test;
}

__global__ void device_test()
{
    using namespace Test;

    assert( test(0,0,0) == 0 );
    assert( test(0,0,1) == 1 );
    assert( test(0,1,1) == 2 );
    assert( test(1,1,1) == 1 );
    assert( test(1,1,0) == 0 );
    assert( test(1,0,0) == -1 );
}

void host_test(CuArrays::ManagedArray3D<int> &test)
{
    assert( test(0,0,0) == 0 );
    assert( test(0,0,1) == 1 );
    assert( test(0,1,1) == 2 );
    assert( test(1,1,1) == 1 );
    assert( test(1,1,0) == 0 );
    assert( test(1,0,0) == -1 );
}

int main()
{
    CuArrays::ManagedArray3D<int> test(Test::test);

    test.malloc(2, 2, 2);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                int a = abs<int>(i - j);
                int b = abs<int>(i - k);
                test(i,j,k) = i + a * (i < j ? 1 : -1)
                                + b * (i < k ? 1 : -1);
            }
        }
    }

    test.copyToDevice();

    // run tests
    host_test(test);
    device_test<<<1,1>>>();

    cudaDeviceSynchronize();
    assert(cudaGetLastError() == cudaSuccess);

    return 0;
}
