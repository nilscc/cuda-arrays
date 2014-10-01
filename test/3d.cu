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
        test(i,i,i) = 1;

    assert(test(0,0,0) == 1);
    assert(test(1,1,1) == 1);

    // test.copyToDevice();
}

int main()
{

    host_test();
    // device_test<<<1,1>>>();

    return 0;
}
