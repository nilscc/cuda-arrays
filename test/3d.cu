#include <CuArrays>
#include <cassert>

namespace Test
{
    __constant__ CuArrays::Array3D<int> test;
}

__global__ void device_test()
{
    using namespace Test;

    assert(test(0,0,0) == 1);
    assert(test(1,1,1) == 1);
}

void host_test()
{
    CuArrays::ManagedArray3D<int> test(Test::test);

    test.malloc(2, 2, 2);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            int d = abs<int>(i - j);
            for (int k = 0; k < 2; k++)
                test(i,j,k) = i + d * (i < j ? 1 : -1);
        }
    }

    test.copyToDevice();

    assert(test(0,0,0) == 0);
    assert(test(1,1,1) == 1);
}

int main()
{
    host_test();
    device_test<<<1,1>>>();

    return 0;
}
