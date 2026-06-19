// realesrgan implemented with ncnn library

#ifndef REALESRGAN_H
#define REALESRGAN_H

#if defined(_WIN32)
  #if defined(REALESRGAN_BUILDING_DLL)
    #define REALESRGAN_API __declspec(dllexport)
  #else
    #define REALESRGAN_API __declspec(dllimport)
  #endif
#else
  #define REALESRGAN_API __attribute__((visibility("default")))
#endif

#include <string>
#include <atomic>

// ncnn
#include "net.h"
#include "gpu.h"
#include "layer.h"

class REALESRGAN_API RealESRGAN
{
public:
    RealESRGAN(int gpuid, bool tta_mode = false);
    ~RealESRGAN();

#if _WIN32
    int load(const std::wstring &parampath, const std::wstring &modelpath);
#else
    int load(const std::string &parampath, const std::string &modelpath);
#endif

    int process(const ncnn::Mat &inimage, ncnn::Mat &outimage, const std::atomic<bool> *abortFlag = nullptr) const;

    // Plain-pointer wrapper: RGBA pixels in -> RGBA pixels out.
    // outPixels must point to a buffer of outW * outH * 4 bytes.
    // Returns 0 on success, non-zero on error.
    int processPixels(const unsigned char *inPixels,  int inW,  int inH,
                      unsigned char       *outPixels, int outW, int outH,
                      const std::atomic<bool> *abortFlag = nullptr) const;

    // Returns the recommended tilesize based on available GPU VRAM.
    // Must be called after construction (Vulkan is already initialised by then).
    int autoTilesize() const;

public:
    // realesrgan parameters
    int scale;
    int tilesize;
    int prepadding;

private:
    ncnn::Net net;
    ncnn::Pipeline *realesrgan_preproc;
    ncnn::Pipeline *realesrgan_postproc;
    ncnn::Layer *bicubic_2x;
    ncnn::Layer *bicubic_3x;
    ncnn::Layer *bicubic_4x;
    bool tta_mode;
};

#endif // REALESRGAN_H
