#include <CuArrays>

using namespace CuArrays;

namespace Device
{
    __constant__ Array3D<double> test;
}

int main()
{
    ManagedArray3D<double> test(Device::test);

    test.malloc(100,100,100);

    test(0,0,0) = 10;

    test.copyToDevice();
    test.copyFromDevice();

    cudaDeviceSynchronize();

    return 0;
}
