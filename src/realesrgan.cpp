#include "realesrgan.h"

#include <algorithm>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <vector>


#if _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace {
constexpr int kModelLoadFailure = -1;
constexpr int kInvalidProcessInputFailure = -2;
constexpr int kAllocatorUnavailableFailure = -3;
constexpr uint64_t kBytesPerMebibyte = 1024ULL * 1024ULL;
constexpr uint64_t kLargeDeviceHeapThresholdBytes =
    4000ULL * kBytesPerMebibyte;
constexpr uint64_t kVeryHighDeviceBudgetBytes =
    5000ULL * kBytesPerMebibyte;
constexpr uint64_t kHighDeviceBudgetBytes = 3000ULL * kBytesPerMebibyte;
constexpr uint64_t kMediumDeviceBudgetBytes = 1900ULL * kBytesPerMebibyte;
constexpr uint64_t kLowDeviceBudgetBytes = 550ULL * kBytesPerMebibyte;
constexpr uint64_t kVeryLowDeviceBudgetBytes = 190ULL * kBytesPerMebibyte;
constexpr int kVeryHighBudgetTileSize = 512;
constexpr int kHighBudgetTileSize = 400;
constexpr int kMediumBudgetTileSize = 200;
constexpr int kLowBudgetTileSize = 100;
constexpr int kVeryLowBudgetTileSize = 64;
constexpr int kMinimumBudgetTileSize = 32;
constexpr int kFallbackTileSize = 100;
constexpr int kIntegratedGpuType = 1;
constexpr int kCpuDeviceType = 3;
constexpr uint64_t kRgbaChannelCount = 4;
constexpr uint64_t kRgbChannelCount = 3;
constexpr uint64_t kPackedRgbaBytesPerPixel = 4;
constexpr uint64_t kFp16BytesPerChannel = sizeof(uint16_t);
constexpr uint64_t kFp32BytesPerChannel = sizeof(float);
constexpr uint64_t kTtaTransformCount = 8;
constexpr uint64_t kSingleTransformCount = 1;
constexpr uint64_t kPaddingSideCount = 2;
constexpr uint64_t kLargeHeapReserveDenominator = 10;
constexpr uint64_t kLargeHeapReservedParts = 3;
constexpr uint64_t kSmallHeapBudgetDivisor = 2;
constexpr uint64_t kWorkspaceConcurrentFeatureMapCount = 4;
constexpr uint64_t kWorkspaceFeatureChannelCount = 64;

// Reserve four simultaneous 64-channel fp16 feature maps at the output-tile
// resolution. Known input/output and staging buffers are added separately.
constexpr uint64_t kNetworkWorkspaceBytesPerOutputPixel =
    kWorkspaceConcurrentFeatureMapCount * kWorkspaceFeatureChannelCount *
    kFp16BytesPerChannel;

std::optional<uint64_t>
checkedProduct(std::initializer_list<uint64_t> factors) {
  uint64_t product = 1;
  for (const uint64_t factor : factors) {
    if (factor != 0 &&
        product > (std::numeric_limits<uint64_t>::max)() / factor) {
      return std::nullopt;
    }
    product *= factor;
  }
  return product;
}

std::optional<uint64_t>
checkedSum(std::initializer_list<uint64_t> terms) {
  uint64_t sum = 0;
  for (const uint64_t term : terms) {
    if (sum > (std::numeric_limits<uint64_t>::max)() - term) {
      return std::nullopt;
    }
    sum += term;
  }
  return sum;
}

using FileHandle = std::unique_ptr<FILE, decltype(&fclose)>;
}

static const uint32_t realesrgan_preproc_spv_data[] = {
#include "realesrgan_preproc.spv.hex.h"
};
static const uint32_t realesrgan_preproc_fp16s_spv_data[] = {
#include "realesrgan_preproc_fp16s.spv.hex.h"
};
static const uint32_t realesrgan_preproc_int8s_spv_data[] = {
#include "realesrgan_preproc_int8s.spv.hex.h"
};
static const uint32_t realesrgan_postproc_spv_data[] = {
#include "realesrgan_postproc.spv.hex.h"
};
static const uint32_t realesrgan_postproc_fp16s_spv_data[] = {
#include "realesrgan_postproc_fp16s.spv.hex.h"
};
static const uint32_t realesrgan_postproc_int8s_spv_data[] = {
#include "realesrgan_postproc_int8s.spv.hex.h"
};

static const uint32_t realesrgan_preproc_tta_spv_data[] = {
#include "realesrgan_preproc_tta.spv.hex.h"
};
static const uint32_t realesrgan_preproc_tta_fp16s_spv_data[] = {
#include "realesrgan_preproc_tta_fp16s.spv.hex.h"
};
static const uint32_t realesrgan_preproc_tta_int8s_spv_data[] = {
#include "realesrgan_preproc_tta_int8s.spv.hex.h"
};
static const uint32_t realesrgan_postproc_tta_spv_data[] = {
#include "realesrgan_postproc_tta.spv.hex.h"
};
static const uint32_t realesrgan_postproc_tta_fp16s_spv_data[] = {
#include "realesrgan_postproc_tta_fp16s.spv.hex.h"
};
static const uint32_t realesrgan_postproc_tta_int8s_spv_data[] = {
#include "realesrgan_postproc_tta_int8s.spv.hex.h"
};

RealESRGAN::RealESRGAN(int gpuid, bool _tta_mode) {
  ncnn::create_gpu_instance();

  int gpu_count = ncnn::get_gpu_count();
  int default_gpu = ncnn::get_default_gpu_index();

#if _WIN32
  std::stringstream ss;
  ss << "[Upscayl DLL] GPU count: " << gpu_count
     << ", default GPU: " << default_gpu << ", requested GPU: " << gpuid
     << "\n";
  OutputDebugStringA(ss.str().c_str());
#endif

  if (gpuid < 0) {
    gpuid = default_gpu;
  }

#if _WIN32
  std::stringstream ss2;
  ss2 << "[Upscayl DLL] Selected GPU device index: " << gpuid << "\n";
  OutputDebugStringA(ss2.str().c_str());
#endif

  if (gpuid >= 0 && gpuid < gpu_count) {
    const ncnn::GpuInfo &gpu_info = ncnn::get_gpu_info(gpuid);
#if _WIN32
    std::stringstream ss3;
    ss3 << "[Upscayl DLL] GPU device name: " << gpu_info.device_name() << "\n";
    OutputDebugStringA(ss3.str().c_str());
#endif
  }

  net.opt.use_vulkan_compute = true;
  net.opt.use_fp16_packed = true;
  net.opt.use_fp16_storage = true;
  net.opt.use_fp16_arithmetic = false;
  net.opt.use_int8_storage = false;
  net.opt.use_int8_arithmetic = false;

  net.set_vulkan_device(gpuid);

  realesrgan_preproc = 0;
  realesrgan_postproc = 0;
  bicubic_2x = 0;
  bicubic_3x = 0;
  bicubic_4x = 0;
  tta_mode = _tta_mode;
}

RealESRGAN::~RealESRGAN() {
  // Clear network before destroying GPU instance
  net.clear();

  // cleanup preprocess and postprocess pipeline
  {
    delete realesrgan_preproc;
    delete realesrgan_postproc;
  }

  if (bicubic_2x) {
    bicubic_2x->destroy_pipeline(net.opt);
    delete bicubic_2x;
  }

  if (bicubic_3x) {
    bicubic_3x->destroy_pipeline(net.opt);
    delete bicubic_3x;
  }

  if (bicubic_4x) {
    bicubic_4x->destroy_pipeline(net.opt);
    delete bicubic_4x;
  }

  ncnn::destroy_gpu_instance();
}

int RealESRGAN::autoTilesize() const {
  const DeviceMemorySnapshot snapshot = getDeviceMemorySnapshot();
  if (!snapshot.valid) {
    return kFallbackTileSize;
  }

  const uint64_t availableBytes =
      snapshot.usageKnown
          ? snapshot.usageBytes < snapshot.budgetBytes
                ? snapshot.budgetBytes - snapshot.usageBytes
                : 0
          : snapshot.budgetBytes;

  if (availableBytes > kVeryHighDeviceBudgetBytes)
    return kVeryHighBudgetTileSize;
  if (availableBytes > kHighDeviceBudgetBytes)
    return kHighBudgetTileSize;
  if (availableBytes > kMediumDeviceBudgetBytes)
    return kMediumBudgetTileSize;
  if (availableBytes > kLowDeviceBudgetBytes)
    return kLowBudgetTileSize;
  if (availableBytes > kVeryLowDeviceBudgetBytes)
    return kVeryLowBudgetTileSize;
  return kMinimumBudgetTileSize;
}

RealESRGAN::DeviceMemorySnapshot
RealESRGAN::getDeviceMemorySnapshot() const {
  DeviceMemorySnapshot snapshot;
  const ncnn::VulkanDevice *device = net.vulkan_device();
  if (!device) {
    return snapshot;
  }

  const ncnn::GpuInfo &info = device->info;
  snapshot.sharesSystemMemory =
      info.type() == kIntegratedGpuType || info.type() == kCpuDeviceType;
  const VkPhysicalDeviceMemoryProperties &memoryProperties =
      info.physical_device_memory_properties();

  uint32_t deviceLocalHeapIndex = memoryProperties.memoryHeapCount;
  for (uint32_t index = 0; index < memoryProperties.memoryHeapCount; ++index) {
    if (memoryProperties.memoryHeaps[index].flags &
        VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      deviceLocalHeapIndex = index;
      break;
    }
  }

  if (deviceLocalHeapIndex >= memoryProperties.memoryHeapCount) {
    return snapshot;
  }

  const uint64_t heapSizeBytes =
      memoryProperties.memoryHeaps[deviceLocalHeapIndex].size;
  if (!info.support_VK_EXT_memory_budget() ||
      !ncnn::vkGetPhysicalDeviceMemoryProperties2KHR) {
    snapshot.budgetBytes =
        heapSizeBytes >= kLargeDeviceHeapThresholdBytes
            ? heapSizeBytes -
                  (heapSizeBytes / kLargeHeapReserveDenominator) *
                      kLargeHeapReservedParts
            : heapSizeBytes / kSmallHeapBudgetDivisor;
    snapshot.valid = snapshot.budgetBytes > 0;
    return snapshot;
  }

  VkPhysicalDeviceMemoryBudgetPropertiesEXT budgetProperties = {};
  budgetProperties.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;

  VkPhysicalDeviceMemoryProperties2KHR queriedProperties = {};
  queriedProperties.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR;
  queriedProperties.pNext = &budgetProperties;

  ncnn::vkGetPhysicalDeviceMemoryProperties2KHR(info.physical_device(),
                                                &queriedProperties);

  snapshot.budgetBytes = budgetProperties.heapBudget[deviceLocalHeapIndex];
  snapshot.usageBytes = budgetProperties.heapUsage[deviceLocalHeapIndex];
  snapshot.usageKnown = true;
  snapshot.valid = snapshot.budgetBytes > 0;
  return snapshot;
}

RealESRGAN::ResourceEstimate
RealESRGAN::estimateResources(const ResourceRequest &request) const {
  ResourceEstimate estimate;
  if (request.inputWidth <= 0 || request.inputHeight <= 0 || scale <= 0 ||
      tilesize <= 0 || prepadding < 0) {
    return estimate;
  }

  const uint64_t inputWidth = static_cast<uint64_t>(request.inputWidth);
  const uint64_t inputHeight = static_cast<uint64_t>(request.inputHeight);
  const uint64_t scaleFactor = static_cast<uint64_t>(scale);
  const uint64_t tileSize = static_cast<uint64_t>(tilesize);
  const uint64_t padding = static_cast<uint64_t>(prepadding);
  const uint64_t transformCount =
      tta_mode ? kTtaTransformCount : kSingleTransformCount;

  const auto doubledPadding =
      checkedProduct({padding, kPaddingSideCount});
  const auto outputWidth = checkedProduct({inputWidth, scaleFactor});
  const auto outputHeight = checkedProduct({inputHeight, scaleFactor});
  if (!doubledPadding || !outputWidth || !outputHeight) {
    return estimate;
  }

  const auto paddedTileWidth =
      checkedSum({(std::min)(inputWidth, tileSize), *doubledPadding});
  const auto paddedTileHeight =
      checkedSum({(std::min)(inputHeight, tileSize), *doubledPadding});
  const auto maximumInputStripeHeight =
      checkedSum({tileSize, *doubledPadding});
  if (!paddedTileWidth || !paddedTileHeight ||
      !maximumInputStripeHeight) {
    return estimate;
  }

  const uint64_t inputStripeHeight =
      (std::min)(inputHeight, *maximumInputStripeHeight);
  const uint64_t activeTileWidth = (std::min)(inputWidth, tileSize);
  const uint64_t activeTileHeight = (std::min)(inputHeight, tileSize);
  const auto outputStripeHeight =
      checkedProduct({activeTileHeight, scaleFactor});
  const auto paddedOutputTileWidth =
      checkedProduct({*paddedTileWidth, scaleFactor});
  const auto paddedOutputTileHeight =
      checkedProduct({*paddedTileHeight, scaleFactor});
  const auto activeOutputTileWidth =
      checkedProduct({activeTileWidth, scaleFactor});
  const auto activeOutputTileHeight =
      checkedProduct({activeTileHeight, scaleFactor});
  if (!outputStripeHeight || !paddedOutputTileWidth ||
      !paddedOutputTileHeight || !activeOutputTileWidth ||
      !activeOutputTileHeight) {
    return estimate;
  }

  const auto inputHostBytes =
      checkedProduct({inputWidth, inputStripeHeight, kRgbaChannelCount,
                      kFp32BytesPerChannel});
  const auto outputHostBytes =
      checkedProduct({*outputWidth, *outputStripeHeight, kRgbaChannelCount,
                      kFp32BytesPerChannel});
  if (!inputHostBytes || !outputHostBytes) {
    return estimate;
  }

  // CPU Mats and mapped Vulkan staging buffers coexist at peak.
  const auto cpuWorkingBytes =
      checkedSum({*inputHostBytes, *outputHostBytes, *inputHostBytes,
                  *outputHostBytes});

  const auto inputTileBytes =
      checkedProduct({*paddedTileWidth, *paddedTileHeight, kRgbChannelCount,
                      kFp16BytesPerChannel, transformCount});
  const auto inputAlphaTileBytes =
      checkedProduct({activeTileWidth, activeTileHeight,
                      kFp16BytesPerChannel});
  const auto outputTileBytes =
      checkedProduct({*paddedOutputTileWidth, *paddedOutputTileHeight,
                      kRgbChannelCount, kFp16BytesPerChannel,
                      transformCount});
  const auto outputAlphaTileBytes =
      checkedProduct({*activeOutputTileWidth, *activeOutputTileHeight,
                      kFp16BytesPerChannel});
  const auto networkWorkspaceBytes =
      checkedProduct({*paddedOutputTileWidth, *paddedOutputTileHeight,
                      kNetworkWorkspaceBytesPerOutputPixel});
  std::optional<uint64_t> deviceWorkingBytes;
  if (inputTileBytes && inputAlphaTileBytes && outputTileBytes &&
      outputAlphaTileBytes && networkWorkspaceBytes) {
    deviceWorkingBytes =
        checkedSum({*inputHostBytes, *outputHostBytes, *inputTileBytes,
                    *inputAlphaTileBytes, *outputTileBytes,
                    *outputAlphaTileBytes, *networkWorkspaceBytes});
  }

  if (!cpuWorkingBytes || !deviceWorkingBytes) {
    return estimate;
  }

  estimate.cpuWorkingBytes = *cpuWorkingBytes;
  estimate.deviceWorkingBytes = *deviceWorkingBytes;
  estimate.valid = true;
  return estimate;
}

#if _WIN32
int RealESRGAN::load(const std::wstring &parampath,
                     const std::wstring &modelpath)
#else
int RealESRGAN::load(const std::string &parampath, const std::string &modelpath)
#endif
{
#if _WIN32
  {
    FileHandle fp(_wfopen(parampath.c_str(), L"rb"), &fclose);
    if (!fp) {
      fwprintf(stderr, L"🚨 Error: Failed to open %ls\n", parampath.c_str());
      return kModelLoadFailure;
    }

    const int result = net.load_param(fp.get());
    if (result != 0) {
      fwprintf(stderr, L"🚨 Error: Failed to load %ls\n", parampath.c_str());
      return result;
    }
  }
  {
    FileHandle fp(_wfopen(modelpath.c_str(), L"rb"), &fclose);
    if (!fp) {
      fwprintf(stderr, L"🚨 Error: Failed to open %ls\n", modelpath.c_str());
      return kModelLoadFailure;
    }

    const int result = net.load_model(fp.get());
    if (result != 0) {
      fwprintf(stderr, L"🚨 Error: Failed to load %ls\n", modelpath.c_str());
      return result;
    }
  }
#else
  const int paramResult = net.load_param(parampath.c_str());
  if (paramResult != 0)
    return paramResult;

  const int modelResult = net.load_model(modelpath.c_str());
  if (modelResult != 0)
    return modelResult;
#endif

  // initialize preprocess and postprocess pipeline
  {
    std::vector<ncnn::vk_specialization_type> specializations(1);
#if _WIN32
    specializations[0].i = 1;
#else
    specializations[0].i = 0;
#endif

    realesrgan_preproc = new ncnn::Pipeline(net.vulkan_device());
    realesrgan_preproc->set_optimal_local_size_xyz(32, 32, 3);

    realesrgan_postproc = new ncnn::Pipeline(net.vulkan_device());
    realesrgan_postproc->set_optimal_local_size_xyz(32, 32, 3);

    if (tta_mode) {
      if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
        realesrgan_preproc->create(
            realesrgan_preproc_tta_int8s_spv_data,
            sizeof(realesrgan_preproc_tta_int8s_spv_data), specializations);
      else if (net.opt.use_fp16_storage)
        realesrgan_preproc->create(
            realesrgan_preproc_tta_fp16s_spv_data,
            sizeof(realesrgan_preproc_tta_fp16s_spv_data), specializations);
      else
        realesrgan_preproc->create(realesrgan_preproc_tta_spv_data,
                                   sizeof(realesrgan_preproc_tta_spv_data),
                                   specializations);

      if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
        realesrgan_postproc->create(
            realesrgan_postproc_tta_int8s_spv_data,
            sizeof(realesrgan_postproc_tta_int8s_spv_data), specializations);
      else if (net.opt.use_fp16_storage)
        realesrgan_postproc->create(
            realesrgan_postproc_tta_fp16s_spv_data,
            sizeof(realesrgan_postproc_tta_fp16s_spv_data), specializations);
      else
        realesrgan_postproc->create(realesrgan_postproc_tta_spv_data,
                                    sizeof(realesrgan_postproc_tta_spv_data),
                                    specializations);
    } else {
      if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
        realesrgan_preproc->create(realesrgan_preproc_int8s_spv_data,
                                   sizeof(realesrgan_preproc_int8s_spv_data),
                                   specializations);
      else if (net.opt.use_fp16_storage)
        realesrgan_preproc->create(realesrgan_preproc_fp16s_spv_data,
                                   sizeof(realesrgan_preproc_fp16s_spv_data),
                                   specializations);
      else
        realesrgan_preproc->create(realesrgan_preproc_spv_data,
                                   sizeof(realesrgan_preproc_spv_data),
                                   specializations);

      if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
        realesrgan_postproc->create(realesrgan_postproc_int8s_spv_data,
                                    sizeof(realesrgan_postproc_int8s_spv_data),
                                    specializations);
      else if (net.opt.use_fp16_storage)
        realesrgan_postproc->create(realesrgan_postproc_fp16s_spv_data,
                                    sizeof(realesrgan_postproc_fp16s_spv_data),
                                    specializations);
      else
        realesrgan_postproc->create(realesrgan_postproc_spv_data,
                                    sizeof(realesrgan_postproc_spv_data),
                                    specializations);
    }
  }

  // bicubic 2x/3x/4x for alpha channel
  {
    bicubic_2x = ncnn::create_layer("Interp");
    bicubic_2x->vkdev = net.vulkan_device();

    ncnn::ParamDict pd;
    pd.set(0, 3); // bicubic
    pd.set(1, 2.f);
    pd.set(2, 2.f);
    bicubic_2x->load_param(pd);

    bicubic_2x->create_pipeline(net.opt);
  }
  {
    bicubic_3x = ncnn::create_layer("Interp");
    bicubic_3x->vkdev = net.vulkan_device();

    ncnn::ParamDict pd;
    pd.set(0, 3); // bicubic
    pd.set(1, 3.f);
    pd.set(2, 3.f);
    bicubic_3x->load_param(pd);

    bicubic_3x->create_pipeline(net.opt);
  }
  {
    bicubic_4x = ncnn::create_layer("Interp");
    bicubic_4x->vkdev = net.vulkan_device();

    ncnn::ParamDict pd;
    pd.set(0, 3); // bicubic
    pd.set(1, 4.f);
    pd.set(2, 4.f);
    bicubic_4x->load_param(pd);

    bicubic_4x->create_pipeline(net.opt);
  }

  return 0;
}

int RealESRGAN::process(const ncnn::Mat &inimage, ncnn::Mat &outimage, const std::atomic<bool> *abortFlag) const {
  const unsigned char *pixeldata = (const unsigned char *)inimage.data;
  const int w = inimage.w;
  const int h = inimage.h;
  const int channels = inimage.elempack;
  std::optional<uint64_t> expectedOutputWidth;
  std::optional<uint64_t> expectedOutputHeight;
  if (w > 0 && h > 0 && scale > 0) {
    expectedOutputWidth =
        checkedProduct({static_cast<uint64_t>(w),
                        static_cast<uint64_t>(scale)});
    expectedOutputHeight =
        checkedProduct({static_cast<uint64_t>(h),
                        static_cast<uint64_t>(scale)});
  }
  if (!pixeldata || !outimage.data || !expectedOutputWidth ||
      !expectedOutputHeight ||
      *expectedOutputWidth != static_cast<uint64_t>(outimage.w) ||
      *expectedOutputHeight != static_cast<uint64_t>(outimage.h) ||
      outimage.elempack != channels ||
      (channels != static_cast<int>(kRgbChannelCount) &&
       channels != static_cast<int>(kRgbaChannelCount)) ||
      tilesize <= 0 || prepadding < 0) {
    return kInvalidProcessInputFailure;
  }
  if (!net.vulkan_device()) {
    return kAllocatorUnavailableFailure;
  }

  const int TILE_SIZE_X = tilesize;
  const int TILE_SIZE_Y = tilesize;

  ncnn::VkAllocator *blob_vkallocator =
      net.vulkan_device()->acquire_blob_allocator();
  ncnn::VkAllocator *staging_vkallocator =
      net.vulkan_device()->acquire_staging_allocator();
  if (!blob_vkallocator || !staging_vkallocator) {
    if (blob_vkallocator) {
      net.vulkan_device()->reclaim_blob_allocator(blob_vkallocator);
    }
    if (staging_vkallocator) {
      net.vulkan_device()->reclaim_staging_allocator(staging_vkallocator);
    }
    return kAllocatorUnavailableFailure;
  }

  ncnn::Option opt = net.opt;
  opt.blob_vkallocator = blob_vkallocator;
  opt.workspace_vkallocator = blob_vkallocator;
  opt.staging_vkallocator = staging_vkallocator;

  // each tile 100x100
  const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
  const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

  const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

  bool aborted = false;

  // #pragma omp parallel for num_threads(2)
  for (int yi = 0; yi < ytiles; yi++) {
    if (abortFlag && abortFlag->load(std::memory_order_relaxed)) {
      aborted = true;
      break;
    }
    const int tile_h_nopad =
        std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;

    int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
    int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding, h);
    const size_t inputTileByteOffset =
        static_cast<size_t>(in_tile_y0) * static_cast<size_t>(w) *
        static_cast<size_t>(channels);
    const unsigned char *inputTileData = pixeldata + inputTileByteOffset;

    ncnn::Mat in;
    if (opt.use_fp16_storage && opt.use_int8_storage) {
      in = ncnn::Mat(w, (in_tile_y1 - in_tile_y0),
                     const_cast<unsigned char *>(inputTileData),
                     static_cast<size_t>(channels), 1);
    } else {
      if (channels == 3) {
#if _WIN32
        in = ncnn::Mat::from_pixels(inputTileData, ncnn::Mat::PIXEL_BGR2RGB, w,
                                    (in_tile_y1 - in_tile_y0));
#else
        in = ncnn::Mat::from_pixels(inputTileData, ncnn::Mat::PIXEL_RGB, w,
                                    (in_tile_y1 - in_tile_y0));
#endif
      }
      if (channels == 4) {
#if _WIN32
        in = ncnn::Mat::from_pixels(inputTileData, ncnn::Mat::PIXEL_BGRA2RGBA, w,
                                    (in_tile_y1 - in_tile_y0));
#else
        in = ncnn::Mat::from_pixels(inputTileData, ncnn::Mat::PIXEL_RGBA, w,
                                    (in_tile_y1 - in_tile_y0));
#endif
      }
    }

    ncnn::VkCompute cmd(net.vulkan_device());

    // upload
    ncnn::VkMat in_gpu;
    {
      cmd.record_clone(in, in_gpu, opt);

      if (xtiles > 1) {
        cmd.submit_and_wait();
        cmd.reset();
      }
    }

    int out_tile_y0 = std::max(yi * TILE_SIZE_Y, 0);
    int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h);

    ncnn::VkMat out_gpu;
    if (opt.use_fp16_storage && opt.use_int8_storage) {
      out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale,
                     (size_t)channels, 1, blob_vkallocator);
    } else {
      out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, channels,
                     (size_t)4u, 1, blob_vkallocator);
    }

    for (int xi = 0; xi < xtiles; xi++) {
      if (abortFlag && abortFlag->load(std::memory_order_relaxed)) {
        aborted = true;
        break;
      }
      const int tile_w_nopad =
          std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;

      if (tta_mode) {
        // preproc
        ncnn::VkMat in_tile_gpu[8];
        ncnn::VkMat in_alpha_tile_gpu;
        {
          // crop tile
          int tile_x0 = xi * TILE_SIZE_X - prepadding;
          int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding;
          int tile_y0 = yi * TILE_SIZE_Y - prepadding;
          int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding;

          in_tile_gpu[0].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                in_out_tile_elemsize, 1, blob_vkallocator);
          in_tile_gpu[1].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                in_out_tile_elemsize, 1, blob_vkallocator);
          in_tile_gpu[2].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                in_out_tile_elemsize, 1, blob_vkallocator);
          in_tile_gpu[3].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                                in_out_tile_elemsize, 1, blob_vkallocator);
          in_tile_gpu[4].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3,
                                in_out_tile_elemsize, 1, blob_vkallocator);
          in_tile_gpu[5].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3,
                                in_out_tile_elemsize, 1, blob_vkallocator);
          in_tile_gpu[6].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3,
                                in_out_tile_elemsize, 1, blob_vkallocator);
          in_tile_gpu[7].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3,
                                in_out_tile_elemsize, 1, blob_vkallocator);

          if (channels == 4) {
            in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1,
                                     in_out_tile_elemsize, 1, blob_vkallocator);
          }

          std::vector<ncnn::VkMat> bindings(10);
          bindings[0] = in_gpu;
          bindings[1] = in_tile_gpu[0];
          bindings[2] = in_tile_gpu[1];
          bindings[3] = in_tile_gpu[2];
          bindings[4] = in_tile_gpu[3];
          bindings[5] = in_tile_gpu[4];
          bindings[6] = in_tile_gpu[5];
          bindings[7] = in_tile_gpu[6];
          bindings[8] = in_tile_gpu[7];
          bindings[9] = in_alpha_tile_gpu;

          std::vector<ncnn::vk_constant_type> constants(13);
          constants[0].i = in_gpu.w;
          constants[1].i = in_gpu.h;
          constants[2].i = in_gpu.cstep;
          constants[3].i = in_tile_gpu[0].w;
          constants[4].i = in_tile_gpu[0].h;
          constants[5].i = in_tile_gpu[0].cstep;
          constants[6].i = prepadding;
          constants[7].i = prepadding;
          constants[8].i = xi * TILE_SIZE_X;
          constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
          constants[10].i = channels;
          constants[11].i = in_alpha_tile_gpu.w;
          constants[12].i = in_alpha_tile_gpu.h;

          ncnn::VkMat dispatcher;
          dispatcher.w = in_tile_gpu[0].w;
          dispatcher.h = in_tile_gpu[0].h;
          dispatcher.c = channels;

          cmd.record_pipeline(realesrgan_preproc, bindings, constants,
                              dispatcher);
        }

        // realesrgan
        ncnn::VkMat out_tile_gpu[8];
        for (int ti = 0; ti < 8; ti++) {
          ncnn::Extractor ex = net.create_extractor();

          ex.set_blob_vkallocator(blob_vkallocator);
          ex.set_workspace_vkallocator(blob_vkallocator);
          ex.set_staging_vkallocator(staging_vkallocator);

          ex.input("data", in_tile_gpu[ti]);

          ex.extract("output", out_tile_gpu[ti], cmd);

          {
            cmd.submit_and_wait();
            cmd.reset();
          }
        }

        ncnn::VkMat out_alpha_tile_gpu;
        if (channels == 4) {
          if (scale == 1) {
            out_alpha_tile_gpu = in_alpha_tile_gpu;
          }
          if (scale == 2) {
            bicubic_2x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd,
                                opt);
          }
          if (scale == 3) {
            bicubic_3x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd,
                                opt);
          }
          if (scale == 4) {
            bicubic_4x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd,
                                opt);
          }
        }

        // postproc
        {
          std::vector<ncnn::VkMat> bindings(10);
          bindings[0] = out_tile_gpu[0];
          bindings[1] = out_tile_gpu[1];
          bindings[2] = out_tile_gpu[2];
          bindings[3] = out_tile_gpu[3];
          bindings[4] = out_tile_gpu[4];
          bindings[5] = out_tile_gpu[5];
          bindings[6] = out_tile_gpu[6];
          bindings[7] = out_tile_gpu[7];
          bindings[8] = out_alpha_tile_gpu;
          bindings[9] = out_gpu;

          std::vector<ncnn::vk_constant_type> constants(13);
          constants[0].i = out_tile_gpu[0].w;
          constants[1].i = out_tile_gpu[0].h;
          constants[2].i = out_tile_gpu[0].cstep;
          constants[3].i = out_gpu.w;
          constants[4].i = out_gpu.h;
          constants[5].i = out_gpu.cstep;
          constants[6].i = xi * TILE_SIZE_X * scale;
          constants[7].i = std::min(TILE_SIZE_X * scale,
                                    out_gpu.w - xi * TILE_SIZE_X * scale);
          constants[8].i = prepadding * scale;
          constants[9].i = prepadding * scale;
          constants[10].i = channels;
          constants[11].i = out_alpha_tile_gpu.w;
          constants[12].i = out_alpha_tile_gpu.h;

          ncnn::VkMat dispatcher;
          dispatcher.w = std::min(TILE_SIZE_X * scale,
                                  out_gpu.w - xi * TILE_SIZE_X * scale);
          dispatcher.h = out_gpu.h;
          dispatcher.c = channels;

          cmd.record_pipeline(realesrgan_postproc, bindings, constants,
                              dispatcher);
        }
      } else {
        // preproc
        ncnn::VkMat in_tile_gpu;
        ncnn::VkMat in_alpha_tile_gpu;
        {
          // crop tile
          int tile_x0 = xi * TILE_SIZE_X - prepadding;
          int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding;
          int tile_y0 = yi * TILE_SIZE_Y - prepadding;
          int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding;

          in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3,
                             in_out_tile_elemsize, 1, blob_vkallocator);

          if (channels == 4) {
            in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1,
                                     in_out_tile_elemsize, 1, blob_vkallocator);
          }

          std::vector<ncnn::VkMat> bindings(3);
          bindings[0] = in_gpu;
          bindings[1] = in_tile_gpu;
          bindings[2] = in_alpha_tile_gpu;

          std::vector<ncnn::vk_constant_type> constants(13);
          constants[0].i = in_gpu.w;
          constants[1].i = in_gpu.h;
          constants[2].i = in_gpu.cstep;
          constants[3].i = in_tile_gpu.w;
          constants[4].i = in_tile_gpu.h;
          constants[5].i = in_tile_gpu.cstep;
          constants[6].i = prepadding;
          constants[7].i = prepadding;
          constants[8].i = xi * TILE_SIZE_X;
          constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
          constants[10].i = channels;
          constants[11].i = in_alpha_tile_gpu.w;
          constants[12].i = in_alpha_tile_gpu.h;

          ncnn::VkMat dispatcher;
          dispatcher.w = in_tile_gpu.w;
          dispatcher.h = in_tile_gpu.h;
          dispatcher.c = channels;

          cmd.record_pipeline(realesrgan_preproc, bindings, constants,
                              dispatcher);
        }

        // realesrgan
        ncnn::VkMat out_tile_gpu;
        {
          ncnn::Extractor ex = net.create_extractor();

          ex.set_blob_vkallocator(blob_vkallocator);
          ex.set_workspace_vkallocator(blob_vkallocator);
          ex.set_staging_vkallocator(staging_vkallocator);

          ex.input("data", in_tile_gpu);

          ex.extract("output", out_tile_gpu, cmd);
        }

        ncnn::VkMat out_alpha_tile_gpu;
        if (channels == 4) {
          if (scale == 1) {
            out_alpha_tile_gpu = in_alpha_tile_gpu;
          }
          if (scale == 2) {
            bicubic_2x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd,
                                opt);
          }
          if (scale == 3) {
            bicubic_3x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd,
                                opt);
          }
          if (scale == 4) {
            bicubic_4x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd,
                                opt);
          }
        }

        // postproc
        {
          std::vector<ncnn::VkMat> bindings(3);
          bindings[0] = out_tile_gpu;
          bindings[1] = out_alpha_tile_gpu;
          bindings[2] = out_gpu;

          std::vector<ncnn::vk_constant_type> constants(13);
          constants[0].i = out_tile_gpu.w;
          constants[1].i = out_tile_gpu.h;
          constants[2].i = out_tile_gpu.cstep;
          constants[3].i = out_gpu.w;
          constants[4].i = out_gpu.h;
          constants[5].i = out_gpu.cstep;
          constants[6].i = xi * TILE_SIZE_X * scale;
          constants[7].i = std::min(TILE_SIZE_X * scale,
                                    out_gpu.w - xi * TILE_SIZE_X * scale);
          constants[8].i = prepadding * scale;
          constants[9].i = prepadding * scale;
          constants[10].i = channels;
          constants[11].i = out_alpha_tile_gpu.w;
          constants[12].i = out_alpha_tile_gpu.h;

          ncnn::VkMat dispatcher;
          dispatcher.w = std::min(TILE_SIZE_X * scale,
                                  out_gpu.w - xi * TILE_SIZE_X * scale);
          dispatcher.h = out_gpu.h;
          dispatcher.c = channels;

          cmd.record_pipeline(realesrgan_postproc, bindings, constants,
                              dispatcher);
        }
      }

      if (xtiles > 1) {
        cmd.submit_and_wait();
        cmd.reset();
      }

      fprintf(stderr, "%.2f%%\n",
              (float)(yi * xtiles + xi) / (ytiles * xtiles) * 100);
    }

    if (aborted) {
      break;
    }

    // download
    {
      ncnn::Mat out;
      const size_t outputTileByteOffset =
          static_cast<size_t>(yi) * static_cast<size_t>(scale) *
          static_cast<size_t>(TILE_SIZE_Y) * static_cast<size_t>(w) *
          static_cast<size_t>(scale) * static_cast<size_t>(channels);
      unsigned char *outputTileData =
          static_cast<unsigned char *>(outimage.data) + outputTileByteOffset;

      if (opt.use_fp16_storage && opt.use_int8_storage) {
        out = ncnn::Mat(out_gpu.w, out_gpu.h, outputTileData,
                        static_cast<size_t>(channels), 1);
      }

      cmd.record_clone(out_gpu, out, opt);

      cmd.submit_and_wait();

      if (!(opt.use_fp16_storage && opt.use_int8_storage)) {
        if (channels == 3) {
#if _WIN32
          out.to_pixels(outputTileData, ncnn::Mat::PIXEL_RGB2BGR);
#else
          out.to_pixels(outputTileData, ncnn::Mat::PIXEL_RGB);
#endif
        }
        if (channels == 4) {
#if _WIN32
          out.to_pixels(outputTileData, ncnn::Mat::PIXEL_RGBA2BGRA);
#else
          out.to_pixels(outputTileData, ncnn::Mat::PIXEL_RGBA);
#endif
        }
      }
    }
  }

  blob_vkallocator->clear();
  staging_vkallocator->clear();
  net.vulkan_device()->reclaim_blob_allocator(blob_vkallocator);
  net.vulkan_device()->reclaim_staging_allocator(staging_vkallocator);

  if (aborted) {
    return -1;
  }
  return 0;
}

int RealESRGAN::processPixels(const unsigned char *inPixels, int inW, int inH,
                              unsigned char *outPixels, int outW,
                              int outH, const std::atomic<bool> *abortFlag) const {
  if (!inPixels || !outPixels || inW <= 0 || inH <= 0 || outW <= 0 ||
      outH <= 0 || scale <= 0) {
    return kInvalidProcessInputFailure;
  }

  const auto expectedOutputWidth =
      checkedProduct({static_cast<uint64_t>(inW),
                      static_cast<uint64_t>(scale)});
  const auto expectedOutputHeight =
      checkedProduct({static_cast<uint64_t>(inH),
                      static_cast<uint64_t>(scale)});
  if (!expectedOutputWidth || !expectedOutputHeight ||
      *expectedOutputWidth != static_cast<uint64_t>(outW) ||
      *expectedOutputHeight != static_cast<uint64_t>(outH)) {
    return kInvalidProcessInputFailure;
  }

  ncnn::Mat inMat(inW, inH, (void *)inPixels,
                  static_cast<size_t>(kPackedRgbaBytesPerPixel),
                  static_cast<int>(kRgbaChannelCount));
  ncnn::Mat outMat(outW, outH, (void *)outPixels,
                   static_cast<size_t>(kPackedRgbaBytesPerPixel),
                   static_cast<int>(kRgbaChannelCount));

  int ret = process(inMat, outMat, abortFlag);
  if (ret != 0)
    return ret;

  return 0;
}
