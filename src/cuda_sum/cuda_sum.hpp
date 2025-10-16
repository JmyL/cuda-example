#pragma once

class CudaSum {
  public:
    CudaSum(int N, const int *data);
    ~CudaSum();

    int compute_sum();

  private:
    int N;
    int *d_input, *d_output;
};