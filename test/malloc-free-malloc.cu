#include <CuArrays>

namespace Device
{
    __constant__ CuArrays::Array<int> test1d;
    __constant__ CuArrays::Array2D<int> test2d;
    __constant__ CuArrays::Array3D<int> test3d;
}

int main()
{
    /*
     * 1D test
     *
     */

    CuArrays::ManagedArray<int> test1d(Device::test1d);

    test1d.malloc(10);
    test1d.free();

    // repeat
    test1d.malloc(10);
    test1d.free();

    /*
     * 2D test
     *
     */

    CuArrays::ManagedArray2D<int> test2d(Device::test2d);

    test2d.malloc(10,10);
    test2d.free();

    test2d.malloc(10,10);
    test2d.free();

    /*
     * 3D test
     *
     */

    CuArrays::ManagedArray3D<int> test3d(Device::test3d);

    test3d.malloc(10,10,10);
    test3d.free();

    test3d.malloc(10,10,10);
    test3d.free();

    /*
     * quit
     *
     */

    cudaDeviceSynchronize();

    return 0;
}
