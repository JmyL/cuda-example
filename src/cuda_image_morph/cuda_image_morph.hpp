#pragma once

class CudaImageMorph {
  public:
    CudaImageMorph(int width, int height, const unsigned char *data);
    ~CudaImageMorph();

    void naive_erode(unsigned char *output_host = nullptr);
    void naive_dilate(unsigned char *output_host = nullptr);
    void shared_erode(unsigned char *output_host = nullptr);
    void shared_dilate(unsigned char *output_host = nullptr);

  private:
    int width, height;
    unsigned char *d_input, *d_output;
};