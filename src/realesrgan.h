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
#include <cstdint>
#include <optional>

// ncnn
#include "net.h"
#include "gpu.h"
#include "layer.h"

class REALESRGAN_API RealESRGAN
{
public:
    struct ResourceRequest final
    {
        int inputWidth = 0;
        int inputHeight = 0;
    };

    struct ResourceEstimate final
    {
        uint64_t cpuWorkingBytes = 0;
        uint64_t deviceWorkingBytes = 0;
        bool valid = false;
    };

    struct DeviceMemorySnapshot final
    {
        uint64_t budgetBytes = 0;
        uint64_t usageBytes = 0;
        bool usageKnown = false;
        bool sharesSystemMemory = false;
        bool valid = false;
    };

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

    // Conservatively estimates peak transient allocations for one request.
    // Persistent model allocations are reflected in the current device usage.
    [[nodiscard]] ResourceEstimate estimateResources(const ResourceRequest &request) const;
    [[nodiscard]] DeviceMemorySnapshot getDeviceMemorySnapshot() const;

    // Classifies an error code returned by process()/processPixels() as a
    // Vulkan device-loss failure (e.g. after a Windows TDR driver reset),
    // as opposed to a model-load, allocation, or input-validation failure.
    // Callers can use this to decide whether rebuilding this RealESRGAN
    // instance and retrying is worthwhile.
    [[nodiscard]] static bool isDeviceLossError(int errorCode) noexcept;

    // Scale factor derived from the loaded model's own graph (via PixelShuffle
    // upscale_factor along the path to the declared output blob). Populated by
    // load() on success; nullopt if the graph could not be walked to a
    // confident answer (e.g. no PixelShuffle on any path to output). Callers
    // should fall back to their own default when this is nullopt.
    [[nodiscard]] std::optional<int> detectedScale() const noexcept { return cachedDetectedScale; }

public:
    // realesrgan parameters
    int scale;
    int tilesize;
    int prepadding;

private:
    [[nodiscard]] std::optional<int> detectScaleFromGraph() const;

    ncnn::Net net;
    ncnn::Pipeline *realesrgan_preproc;
    ncnn::Pipeline *realesrgan_postproc;
    ncnn::Layer *bicubic_2x;
    ncnn::Layer *bicubic_3x;
    ncnn::Layer *bicubic_4x;
    bool tta_mode;
    std::optional<int> cachedDetectedScale;
};

#endif // REALESRGAN_H
