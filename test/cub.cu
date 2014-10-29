#include <cassert>

#include <cub/device/device_reduce.cuh>

#include <CuArrays>
#include <CuArrays_H/cuda_utils.h>

const double zero = 0;

namespace Red
{
    // number of samples
    const int N = 10;

    // temp storage for reduce
    void *d_tmp = NULL;
    size_t s_tmp = 0;

    // double valued output
    double *d_double_out;
}

namespace Device
{
    __constant__ CuArrays::Array<double> test1d;
    __constant__ CuArrays::Array2D<double> test2d;
    __constant__ CuArrays::Array3D<double> test3d;
}

double sum(double* d_in, size_t N)
{
    using namespace Red;

    cub::DeviceReduce::Reduce(
        d_tmp, s_tmp,
        d_in, d_double_out,
        N,
        cub::Sum()
    );

    // copy result back
    double out;
    cudaVerify( cudaMemcpy(&out, d_double_out, sizeof(double), cudaMemcpyDeviceToHost) );

    printf("sum = %.2e\n", out);

    return out;
}

int main()
{
    CuArrays::ManagedArray<double> test1d(Device::test1d);
    CuArrays::ManagedArray2D<double> test2d(Device::test2d);
    CuArrays::ManagedArray3D<double> test3d(Device::test3d);

    // malloc device space
    test1d.malloc(Red::N);
    test2d.malloc(Red::N,2);
    test3d.malloc(Red::N,2,2);

    // allocate memory for Reduce

    cudaVerify( cudaMalloc(&Red::d_double_out, sizeof(double)) );
    cudaVerify( cudaMemcpy(Red::d_double_out, &zero, sizeof(double), cudaMemcpyHostToDevice) );

    double *d_in  = NULL,
           *d_out = NULL;

    cub::DeviceReduce::Reduce(
        Red::d_tmp, Red::s_tmp,
        d_in, d_out,
        Red::N,
        cub::Sum(),
        0,
        true
    );

    cudaVerify( cudaMalloc(&Red::d_tmp, Red::s_tmp) );

    /*
     * 1D test
     *
     */

    // fill data
    for (int i = 0; i < test1d.N(); i++)
        test1d(i) = i + 0.5;

    test1d.copyToDevice();

    assert( 50 == sum(test1d.devicePtr(), test1d.N()) );
    assert( 50 == sum(test1d.devicePtr(), test1d.N()) );

    /*
     * 2D test
     *
     */

    // fill data
    for (int i = 0; i < test2d.N(); i++)
        for (int j = 0; j < test2d.M(); j++)
            test2d(i,j) = i + 0.5;

    test2d.copyToDevice();

    // sum over each row
    for (int j = 0; j < test2d.M(); j++)
    {
        assert( 50 == sum(test2d.devicePtrAt(0, j), test2d.N()) );
        assert( 50 == sum(test2d.devicePtrAt(0, j), test2d.N()) );
    }

    /*
     * 3D test
     *
     */

    // fill data
    for (int i = 0; i < test3d.N(); i++)
        for (int j = 0; j < test3d.M(); j++)
            for (int k = 0; k < test3d.O(); k++)
                test3d(i,j,k) = i + 0.5;

    test3d.copyToDevice();

    for (int j = 0; j < test3d.M(); j++)
    {
        for (int k = 0; k < test3d.O(); k++)
        {
            assert( 50 == sum(test3d.devicePtrAt(0, j, k), test3d.N()) );
            assert( 50 == sum(test3d.devicePtrAt(0, j, k), test3d.N()) );
        }
    }
}
