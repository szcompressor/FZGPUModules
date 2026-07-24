#pragma once

/**
 * @file include/backend/api.h
 * @brief Backend-neutral spelling of the host-side GPU runtime API.
 *
 * Phase 0 abstracted the GPU *handle types* (`fz::stream_t` and friends, see
 * backend/types.h) but left the ~380 host-side runtime *calls* spelled with
 * their CUDA names (`cudaMalloc`, `cudaMemcpyAsync`, `cudaMemPoolCreate`,
 * `cudaGraphLaunch`, ...). HIP is deliberately source-compatible with the CUDA
 * runtime — its whole design premise is that `hipify` is a textual `cuda`→`hip`
 * rename — so rather than churn every call site, this header re-points the CUDA
 * spellings at their HIP equivalents when the HIP backend is selected.
 *
 * Under the CUDA backend this header expands to nothing at all: no macros are
 * defined, the real CUDA runtime declarations are used unchanged, and the CUDA
 * build is bit-for-bit unaffected.
 *
 * Of the 95 distinct CUDA identifiers used across the non-vendored tree, 90
 * differ only by the `cuda`→`hip` prefix; the five genuine naming divergences
 * are listed explicitly in the "hand-mapped" section below. Anything not
 * covered here simply fails to compile under HIP with an undeclared-identifier
 * error, which is the desired behaviour — a silent fallback would be worse.
 *
 * Include this (rather than <cuda_runtime.h>) from any file that calls the GPU
 * runtime directly. backend/types.h includes it, so most files pick it up
 * transitively via cuda_check.h / mem/mempool.h / stage/stage.h.
 *
 * Deliberately depends on no other project header — it sits at the bottom of
 * the include graph so that the macros below are always established before any
 * call site is parsed.
 */

#if defined(FZGMOD_BACKEND_HIP)

// The real HIP declarations must be visible *before* the macros below are
// defined, so that the macro bodies name already-declared entities.
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// ── Hand-mapped: the five identifiers where HIP did not keep CUDA's spelling.
// Four are naming-convention drift (`DevAttr` → `DeviceAttribute`), one is a
// `_t` suffix CUDA omits.
#define cudaDeviceProp                            hipDeviceProp_t
#define cudaPointerAttributes                     hipPointerAttribute_t
#define cudaDevAttrMaxSharedMemoryPerBlock        hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaDevAttrMaxSharedMemoryPerBlockOptin   hipDeviceAttributeSharedMemPerBlockOptin
#define cudaDevAttrMultiProcessorCount            hipDeviceAttributeMultiprocessorCount

// ── Handle types
#define cudaError_t                 hipError_t
#define cudaStream_t                hipStream_t
#define cudaEvent_t                 hipEvent_t
#define cudaGraph_t                 hipGraph_t
#define cudaGraphExec_t             hipGraphExec_t
#define cudaMemPool_t               hipMemPool_t
#define cudaMemPoolProps            hipMemPoolProps
#define cudaFuncAttribute           hipFuncAttribute
#define cudaStreamCaptureStatus     hipStreamCaptureStatus

// ── Error codes and error reporting
#define cudaSuccess                         hipSuccess
#define cudaErrorInvalidValue               hipErrorInvalidValue
#define cudaErrorNoDevice                   hipErrorNoDevice
#define cudaErrorStreamCaptureUnsupported   hipErrorStreamCaptureUnsupported
#define cudaGetErrorString                  hipGetErrorString
#define cudaGetErrorName                    hipGetErrorName
#define cudaGetLastError                    hipGetLastError

// ── Device management
#define cudaSetDevice                                   hipSetDevice
#define cudaGetDevice                                   hipGetDevice
#define cudaGetDeviceCount                              hipGetDeviceCount
#define cudaGetDeviceProperties                         hipGetDeviceProperties
#define cudaDeviceGetAttribute                          hipDeviceGetAttribute
#define cudaDeviceSynchronize                           hipDeviceSynchronize
#define cudaFuncSetAttribute                            hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize     hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor   hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaProfilerStart                               hipProfilerStart

// ── Plain allocation / transfer
#define cudaMalloc                  hipMalloc
#define cudaFree                    hipFree
#define cudaMallocHost              hipHostMalloc
#define cudaHostAlloc               hipHostAlloc
#define cudaHostAllocDefault        hipHostAllocDefault
#define cudaFreeHost                hipHostFree
#define cudaMemcpy                  hipMemcpy
#define cudaMemcpyAsync             hipMemcpyAsync
#define cudaMemcpyHostToDevice      hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost      hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice    hipMemcpyDeviceToDevice
#define cudaMemset                  hipMemset
#define cudaMemsetAsync             hipMemsetAsync
#define cudaPointerGetAttributes    hipPointerGetAttributes
#define cudaMemoryTypeDevice        hipMemoryTypeDevice
#define cudaMemoryTypeHost          hipMemoryTypeHost

// ── Stream-ordered allocation and memory pools
#define cudaMallocAsync                             hipMallocAsync
#define cudaFreeAsync                               hipFreeAsync
#define cudaMallocFromPoolAsync                     hipMallocFromPoolAsync
#define cudaMemPoolCreate                           hipMemPoolCreate
#define cudaMemPoolDestroy                          hipMemPoolDestroy
#define cudaMemPoolGetAttribute                     hipMemPoolGetAttribute
#define cudaMemPoolSetAttribute                     hipMemPoolSetAttribute
#define cudaMemPoolTrimTo                           hipMemPoolTrimTo
#define cudaMemPoolAttrReleaseThreshold             hipMemPoolAttrReleaseThreshold
#define cudaMemPoolAttrReservedMemCurrent           hipMemPoolAttrReservedMemCurrent
#define cudaMemPoolAttrReservedMemHigh              hipMemPoolAttrReservedMemHigh
#define cudaMemPoolAttrUsedMemCurrent               hipMemPoolAttrUsedMemCurrent
#define cudaMemPoolAttrUsedMemHigh                  hipMemPoolAttrUsedMemHigh
#define cudaMemPoolReuseAllowOpportunistic          hipMemPoolReuseAllowOpportunistic
#define cudaMemPoolReuseAllowInternalDependencies   hipMemPoolReuseAllowInternalDependencies
#define cudaMemPoolReuseFollowEventDependencies     hipMemPoolReuseFollowEventDependencies
#define cudaMemAllocationTypePinned                 hipMemAllocationTypePinned
#define cudaMemLocationTypeDevice                   hipMemLocationTypeDevice
#define cudaMemHandleTypeNone                       hipMemHandleTypeNone

// ── Streams
#define cudaStreamCreate            hipStreamCreate
#define cudaStreamCreateWithFlags   hipStreamCreateWithFlags
#define cudaStreamDestroy           hipStreamDestroy
#define cudaStreamSynchronize       hipStreamSynchronize
#define cudaStreamWaitEvent         hipStreamWaitEvent
#define cudaStreamDefault           hipStreamDefault
#define cudaStreamNonBlocking       hipStreamNonBlocking
#define cudaStreamLegacy            hipStreamLegacy

// ── Events
#define cudaEventCreate             hipEventCreate
#define cudaEventCreateWithFlags    hipEventCreateWithFlags
#define cudaEventDestroy            hipEventDestroy
#define cudaEventRecord             hipEventRecord
#define cudaEventSynchronize        hipEventSynchronize
#define cudaEventElapsedTime        hipEventElapsedTime
#define cudaEventDefault            hipEventDefault
#define cudaEventDisableTiming      hipEventDisableTiming

// ── Graph capture
#define cudaStreamBeginCapture      hipStreamBeginCapture
#define cudaStreamEndCapture        hipStreamEndCapture
#define cudaStreamIsCapturing       hipStreamIsCapturing
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaGraphInstantiate        hipGraphInstantiate
#define cudaGraphLaunch             hipGraphLaunch
#define cudaGraphDestroy            hipGraphDestroy
#define cudaGraphExecDestroy        hipGraphExecDestroy

// Device-side unconditional abort. CUDA exposes this as the named intrinsic
// __trap(); HIP's clang only provides the compiler builtin __builtin_trap().
#define __trap() __builtin_trap()

#elif defined(FZGMOD_BACKEND_SYCL)

#error "FZGMOD_BACKEND=SYCL is not implemented yet"

#else // FZGMOD_BACKEND_CUDA (default)

#include <cuda_runtime.h>

#endif
