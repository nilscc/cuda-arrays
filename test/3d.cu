#include <CuArrays>
#include <cassert>

namespace Test
{
    __constant__ CuArrays::Array3D<int> test;
}

void host_test()
{
    CuArrays::ManagedArray3D<int> test(Test::test);

    test.mallocHost(2, 2, 2);

    for (int i = 0; i < 2; i++)
        test(i,i,i) = 1;

    assert(test(0,0,0) == 1);
    assert(test(1,1,1) == 1);
}

int main()
{
    host_test();

    return 0;
}
