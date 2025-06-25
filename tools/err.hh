#ifndef AE6DCA2E_F19B_41DB_80CB_11230E548F92
#define AE6DCA2E_F19B_41DB_80CB_11230E548F92

#include <cuda_runtime.h>

#include <exception>
#include <sstream>

namespace _portable::utils {

struct exception_gpu_general : public std::exception {
  exception_gpu_general(cudaError_t gpu_error_status, const char* _file_,
                        const int _line_)
      : err_msg([gpu_error_status, _file_, _line_]() {
          const char* err = cudaGetErrorString(gpu_error_status);
          std::stringstream ss;
          // Use standard escape sequences or Unicode
          ss << "GPU API failed at \033[31m\033[1m" << _file_ << ':' << _line_;
          ss << "\033[0m with error: " << err << '(' << (int)gpu_error_status << ')';
          return ss.str();
        }()) {}
  
  const char* what() const noexcept { return err_msg.c_str(); }
  std::string err_msg;
};

}  // namespace _portable::utils

// proxy: not safe to put throw inside a macro expansion
inline void throw_exception_gpu_general(cudaError_t GPU_ERROR_CODE,
                                        const char* _file_, const int _line_) {
  if (cudaSuccess != GPU_ERROR_CODE) {
    throw _portable::utils::exception_gpu_general(GPU_ERROR_CODE, _file_,
                                                  _line_);
  }
}

#define CHECK_GPU(GPU_ERROR_CODE) \
  (throw_exception_gpu_general(GPU_ERROR_CODE, __FILE__, __LINE__))

#define AD_HOC_CHECK_GPU_WITH_LINE(GPU_ERROR_CODE, FILE, LINE) \
  (throw_exception_gpu_general(GPU_ERROR_CODE, FILE, LINE))

#endif /* AE6DCA2E_F19B_41DB_80CB_11230E548F92 */