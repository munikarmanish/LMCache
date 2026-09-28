// SPDX-License-Identifier: Apache-2.0
//
// CPU-side read-bandwidth probe for a CXL DAX pool.
//
// Measures what the CPU can pull out of /dev/dax0.0, which MLC cannot do: MLC
// only sees the pool once it is reconfigured to system-ram (a NUMA node), and
// even then it allocates from the node's low addresses, so on a pool built by
// CONCATENATING two memory modules it never reaches past the first module and
// reports that single module's bandwidth as if it were the whole pool.
//
// This probe addresses the mapping directly, so it can aim threads at one
// module or spread them across both:
//
//   split=0  every thread reads inside module A  -> one module's bandwidth
//   split=1  threads alternate A / B             -> the pool's bandwidth
//
// Comparing the two is how you tell a per-module limit from a pool-wide one.
//
// Cache discipline: each thread streams a large window that is disjoint from
// every other thread's, and reads it with non-temporal loads
// (_mm512_stream_load_si512), so results are not inflated by LLC hits. On a
// host with a very large LLC (this one has 504 MiB) a small or shared buffer
// would otherwise measure cache, not media.
//
// Build:
//   gcc -O3 -mavx512f -pthread -o dax_read scripts/cxl/dax_read.c
//
// Usage:
//   ./dax_read [dev] [threads] [seconds] [split] [module_gib]
//
//   dev         DAX device (default /dev/dax0.0)
//   threads     reader threads (default 8)
//   seconds     measurement duration (default 5)
//   split       0 = all threads on module A, 1 = alternate A/B (default 0)
//   module_gib  size of one module in GiB, i.e. the offset of module B
//               (default 128)
//
// Examples:
//   ./dax_read /dev/dax0.0 16 4 0        # one module
//   ./dax_read /dev/dax0.0 16 4 1 128    # both modules
//
// Note: /dev/dax0.0 must be in devdax mode (daxctl reconfigure-device
// --mode=devdax dax0.0), and the process needs read access to it.

#define _GNU_SOURCE
#include <fcntl.h>
#include <immintrin.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define MAX_THREADS 256

static char *g_base;
static size_t g_span;   // bytes each thread streams, disjoint per thread
static size_t g_gap;    // byte offset of module B (0 when split is off)
static int g_nthr, g_secs, g_split;
static volatile int g_go = 0, g_stop = 0;
static unsigned long long g_bytes[MAX_THREADS];

static void *worker(void *arg) {
    long id = (long)arg;
    // With split on, even threads take module A and odd threads module B, and
    // each side numbers its threads independently so their windows are packed
    // from the start of that module rather than leaving holes.
    size_t side_base = (g_split && (id & 1)) ? g_gap : 0;
    size_t slot = g_split ? (size_t)(id / 2) : (size_t)id;
    char *base = g_base + side_base + slot * g_span;

    unsigned long long n = 0;
    __m512i acc = _mm512_setzero_si512();
    while (!g_go)
        ;
    while (!g_stop) {
        for (size_t off = 0; off + 4096 <= g_span; off += 4096) {
            const char *p = base + off;
            for (int k = 0; k < 4096; k += 64)
                acc = _mm512_add_epi64(acc,
                                       _mm512_stream_load_si512((void *)(p + k)));
            n += 4096;
            if (g_stop)
                break;
        }
    }
    // Keep the loads from being optimised away without perturbing the timing.
    volatile long long sink = _mm512_reduce_add_epi64(acc);
    (void)sink;
    g_bytes[id] = n;
    return NULL;
}

int main(int argc, char **argv) {
    const char *dev = argc > 1 ? argv[1] : "/dev/dax0.0";
    g_nthr = argc > 2 ? atoi(argv[2]) : 8;
    g_secs = argc > 3 ? atoi(argv[3]) : 5;
    g_split = argc > 4 ? atoi(argv[4]) : 0;
    size_t module_gib = argc > 5 ? (size_t)atoll(argv[5]) : 128;

    if (g_nthr < 1 || g_nthr > MAX_THREADS) {
        fprintf(stderr, "threads must be 1..%d\n", MAX_THREADS);
        return 1;
    }
    g_gap = g_split ? (module_gib << 30) : 0;

    int fd = open(dev, O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    // DAX chardevs report st_size 0, so take the extent from sysfs when we can
    // and fall back to two modules' worth.
    size_t maplen = 0;
    {
        const char *slash = strrchr(dev, '/');
        char path[256];
        snprintf(path, sizeof(path), "/sys/bus/dax/devices/%s/size",
                 slash ? slash + 1 : dev);
        FILE *f = fopen(path, "r");
        if (f) {
            unsigned long long v = 0;
            if (fscanf(f, "%llu", &v) == 1)
                maplen = (size_t)v;
            fclose(f);
        }
    }
    if (maplen == 0)
        maplen = (module_gib << 30) * 2;

    g_base = mmap(NULL, maplen, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (g_base == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    // Size each thread's window so the set of them fits in the region actually
    // available. Without this the windows can run past the end of the mapping
    // (a segfault) or, worse, spill silently out of module A into module B and
    // quietly turn a single-module run into a split one.
    // One module's worth in BOTH modes. With split off the point is to measure
    // a single module, so the threads must stay inside it: sizing their
    // windows against the whole mapping would let them spill past the module
    // boundary and silently turn the run into a split one (which reads as a
    // suspiciously high single-module number).
    size_t per_side = module_gib << 30;
    if (per_side > maplen)
        per_side = maplen;
    int threads_per_side = g_split ? (g_nthr + 1) / 2 : g_nthr;
    if (g_split && (module_gib << 30) * 2 > maplen) {
        fprintf(stderr, "module_gib %zu too large for a %zu GiB mapping\n",
                module_gib, maplen >> 30);
        return 1;
    }
    g_span = per_side / (size_t)threads_per_side;
    g_span &= ~((size_t)4096 - 1);
    if (g_span < (1UL << 20)) {
        fprintf(stderr, "too many threads for the available range\n");
        return 1;
    }
    // Cap the window: past a few GiB per thread there is no more cache benefit
    // to defeat, and a smaller footprint keeps first-touch faulting cheap.
    if (g_span > (8UL << 30))
        g_span = 8UL << 30;

    pthread_t th[MAX_THREADS];
    for (long i = 0; i < g_nthr; i++) {
        if (pthread_create(&th[i], NULL, worker, (void *)i) != 0) {
            perror("pthread_create");
            return 1;
        }
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    g_go = 1;
    sleep(g_secs);
    g_stop = 1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    for (int i = 0; i < g_nthr; i++)
        pthread_join(th[i], NULL);

    double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    unsigned long long tot = 0;
    for (int i = 0; i < g_nthr; i++)
        tot += g_bytes[i];
    printf("threads=%-3d split=%d window=%zu MiB  %.2f GB/s\n", g_nthr, g_split,
           g_span >> 20, tot / sec / 1e9);
    return 0;
}
