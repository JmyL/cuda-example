#pragma once

class CudaImageMorph {
  public:
    CudaImageMorph(int width, int height, const unsigned char *data);
    ~CudaImageMorph();

    void erode(unsigned char *output_host);
    void dilate(unsigned char *output_host);

  private:
    int width, height;
    unsigned char *d_input, *d_output;
};