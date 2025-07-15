#include <cuda_runtime.h>
#include <sys/time.h>

#include <cuda/experimental/stf.cuh>
#include <fstream>
#include <iostream>
#include <vector>

using namespace cuda::experimental::stf;

double X0(int i) {
  return sin((double) i);
}

double Y0(int i) {
  return cos((double) i);
}

int main(int argc, char** argv) {
  
  context ctx = stream_ctx();

  const size_t N = 16;
  double X[N], Y[N];

  for (size_t i = 0; i < N; i++)
  {
    X[i] = X0(i);
    Y[i] = Y0(i);
  }

  double alpha = 3.14;

  auto lX = ctx.logical_data(X);
  auto lY = ctx.logical_data(Y);

  float ms_total = 0;
  cudaEvent_t start, stop;

  cuda_safe_call(cudaEventCreate(&start));
  cuda_safe_call(cudaEventCreate(&stop));

  cuda_safe_call(cudaEventRecord(start, ctx.fence()));

  /* Compute Y = Y + alpha X */
  ctx.parallel_for(lY.shape(), lX.read(), lY.rw())->*[alpha] __device__(size_t i, auto dX, auto dY) {
    dY(i) += alpha * dX(i);
  };

  cuda_safe_call(cudaEventRecord(stop, ctx.fence()));


  ctx.finalize();

  cuda_safe_call(cudaEventElapsedTime(&ms_total, start, stop));
  printf("Total time: %.6f ms\n", ms_total);

  for (size_t i = 0; i < N; i++) {
    assert(fabs(Y[i] - (Y0(i) + alpha * X0(i))) < 0.0001);
    assert(fabs(X[i] - X0(i)) < 0.0001);
  }

  return 0;
}