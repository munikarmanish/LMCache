/* SPDX-License-Identifier: Apache-2.0
 *
 * Fast DRAM -> CXL copy for the cross-node PushKVToCXL donor path.
 * Compiled into a small .so loaded via ctypes — see the loader in
 * cross_node.py.
 *
 * Why this exists: the donor writes a chunk's KV bytes from local DRAM
 * into the shared CXL pool. A plain `memmove` (regular stores) pulls each
 * destination cacheline into cache before writing it (read-for-ownership)
 * — wasteful for write-only traffic into device memory. On a real CXL
 * device this measured ~2 GB/s. Non-temporal streaming stores
 * (`_mm256_stream`) bypass the cache and write straight through, measured
 * at ~10.6 GB/s on the same device (a ~5x speedup, matching the device's
 * DMA write bandwidth).
 *
 * The copy is followed by SFENCE so the streaming stores are globally
 * ordered before any subsequent CLFLUSH/MFENCE the caller issues to
 * publish the slot metadata (non-temporal stores are weakly ordered).
 *
 * Build: cc -O3 -march=native -shared -fPIC cxl_copy.c -o cxl_copy.so
 * (the loader in cross_node.py handles this automatically; falls back to
 * memmove when the .so is unavailable or on non-x86.)
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define CXL_COPY_HAS_NT 1
#else
#define CXL_COPY_HAS_NT 0
#endif

/* Returns 1 if non-temporal streaming stores are available, 0 otherwise.
 * The Python loader uses this to decide whether to use this helper or
 * fall back to memmove. */
int cxl_copy_has_nt(void) {
    return CXL_COPY_HAS_NT;
}

/* Copy `n` bytes from `src` (DRAM) to `dst` (CXL pool) using non-temporal
 * 32-byte streaming stores, then SFENCE. On non-x86 this is a plain memcpy.
 *
 * IMPORTANT: `_mm256_stream_si256` (VMOVNTDQ) FAULTS on a destination that
 * is not 32-byte aligned — it is not merely slower. So we scalar-copy a
 * head up to the next 32-byte boundary of `dst`, stream the aligned
 * middle, then scalar-copy the tail. The SOURCE may be unaligned (we use
 * `loadu`). Any (dst, src, n) is therefore safe; alignment only affects
 * how much of the copy takes the fast streaming path.
 */
void cxl_nt_copy(void *dst, const void *src, size_t n) {
#if CXL_COPY_HAS_NT
    char *d = (char *)dst;
    const char *s = (const char *)src;
    size_t i = 0;

    /* Scalar head: advance until `d + i` is 32-byte aligned (or we run
     * out of bytes). ((-addr) & 31) is the bytes to the next boundary. */
    size_t head = ((size_t)(-(uintptr_t)d)) & (size_t)31;
    if (head > n) head = n;
    for (; i < head; ++i) {
        d[i] = s[i];
    }

    /* Streaming middle: dst is now 32-byte aligned. */
    size_t aligned_end = i + ((n - i) & ~(size_t)31);
    for (; i < aligned_end; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(s + i));
        _mm256_stream_si256((__m256i *)(d + i), v);
    }

    /* Scalar tail. */
    for (; i < n; ++i) {
        d[i] = s[i];
    }

    /* Order the non-temporal stores before any later store/flush. */
    _mm_sfence();
#else
    memcpy(dst, src, n);
#endif
}
