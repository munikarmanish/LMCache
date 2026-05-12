/* SPDX-License-Identifier: Apache-2.0
 *
 * Cross-host cacheline visibility primitives for CXL 2.0 shared
 * memory on x86. Compiled into a small .so loaded via ctypes — see
 * fence.py for the loader and CLFlushFence wrapper.
 *
 * On CXL hardware that does NOT enforce cross-host cache coherence
 * (the common case), peers must:
 *   - flush their writes out of local caches so they reach the CXL
 *     device (CLFLUSH on the writer)
 *   - invalidate their stale reads so the next load fetches from
 *     the device (CLFLUSH on the reader)
 *
 * We use CLFLUSH (not CLFLUSHOPT) because CLFLUSH is strongly ordered
 * with respect to surrounding loads and stores. CLFLUSHOPT is faster
 * but may sit in the store buffer past a lock release, which can let
 * a peer that just acquired the lock observe stale metadata.
 *
 * MFENCE provides a full memory barrier; we issue it after a flush
 * sequence to ensure the flushed lines reach the device before any
 * subsequent visible operation (e.g. releasing a lock that grants
 * another node permission to read).
 *
 * Build: gcc -O2 -march=native -shared -fPIC cxl_fence.c -o cxl_fence.so
 * (the loader in fence.py handles this automatically.)
 */

#include <stddef.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#define CXL_FENCE_HAS_CLFLUSH 1
#else
#define CXL_FENCE_HAS_CLFLUSH 0
#endif

/* Cacheline size. 64 bytes is correct for every x86 CPU we expect
 * to deploy on; if a future platform reports something different via
 * cpuid we can detect it dynamically, but 64 is a fine compile-time
 * constant for now. */
static const size_t CXL_CACHELINE = 64;

/* Returns 1 if CLFLUSH is available, 0 otherwise. The Python loader
 * uses this to decide whether to fall back to StubFence. */
int cxl_fence_has_clflush(void) {
    return CXL_FENCE_HAS_CLFLUSH;
}

/* Flush every cacheline that overlaps [addr, addr+size). Idempotent
 * and safe for size==0. */
void cxl_flush_range(void *addr, size_t size) {
#if CXL_FENCE_HAS_CLFLUSH
    if (size == 0 || addr == NULL) return;
    uintptr_t start = (uintptr_t)addr & ~(CXL_CACHELINE - 1);
    uintptr_t end = (uintptr_t)addr + size;
    for (uintptr_t p = start; p < end; p += CXL_CACHELINE) {
        _mm_clflush((const void *)p);
    }
#else
    (void)addr;
    (void)size;
#endif
}

/* Full memory barrier. Combine with cxl_flush_range to publish
 * writes (flush + mfence) or to invalidate stale reads (flush +
 * mfence + load). */
void cxl_mfence(void) {
#if CXL_FENCE_HAS_CLFLUSH
    _mm_mfence();
#endif
}

/* Convenience: flush + mfence in one call. Saves one Python -> C
 * crossing on the writer's "publish" path. */
void cxl_flush_range_and_fence(void *addr, size_t size) {
    cxl_flush_range(addr, size);
    cxl_mfence();
}
