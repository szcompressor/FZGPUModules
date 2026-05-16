// Adapted from PHF reference (origin/v1.1.0_dev:modules/codec/huffman/hf.h)
// Changes: removed C management API (phf_codec, phf_create/release/buildbook/encode/decode),
//          removed pszanalysis_hf_buildtree; kept types, constants, and helper declarations.

#ifndef PHF_HF_H
#define PHF_HF_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

typedef void* phf_stream_t;

#define PHF_SUCCESS 0
#define PHF_WRONG_DTYPE 1
#define PHF_FAIL_GPU_MALLOC 2
#define PHF_FAIL_GPU_MEMCPY 3
#define PHF_FAIL_GPU_ILLEGAL_ACCESS 4
#define PHF_FAIL_GPU_OUT_OF_MEMORY 5
#define PHF_NOT_IMPLEMENTED 99

typedef enum { HF_U1, HF_U2, HF_U4, HF_U8, HF_ULL, HF_INVALID } phf_dtype;

#define PHFHEADER_FORCED_ALIGN 128
#define PHFHEADER_HEADER 0
#define PHFHEADER_REVBK 1
#define PHFHEADER_PAR_NBIT 2
#define PHFHEADER_PAR_ENTRY 3
#define PHFHEADER_BITSTREAM 4
#define PHFHEADER_END 5

typedef uint32_t PHF_METADATA;
typedef uint8_t  PHF_BIN;
typedef uint8_t  PHF_BYTE;

typedef struct {
    int      bklen : 16;
    int      sublen, pardeg;
    size_t   original_len;
    size_t   total_nbit, total_ncell;
    uint32_t entry[PHFHEADER_END + 1];
} phf_header;

// phf_encoded_bytes: total byte length of a PHF-encoded bitstream
#define phf_encoded_bytes capi_phf_encoded_bytes
uint32_t capi_phf_encoded_bytes(phf_header* h);

// phf_reverse_book_bytes: byte size of the reverse codebook for a given bklen
#define phf_reverse_book_bytes capi_phf_reverse_book_bytes
size_t capi_phf_reverse_book_bytes(uint16_t bklen, size_t BK_UNIT_BYTES, size_t SYM_BYTES);

// Partition tuning helpers used by Buf<E> constructor
size_t capi_phf_coarse_tune_sublen(size_t inlen);
void   capi_phf_coarse_tune(size_t len, int* sublen, int* pardeg);

#ifdef __cplusplus
}
#endif
#endif /* PHF_HF_H */
