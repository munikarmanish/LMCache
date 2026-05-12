/* SPDX-License-Identifier: Apache-2.0
 *
 * Standalone CXL lock-manager arbiter.
 *
 * Replaces the Python `LockManager` thread when running in production:
 * the Python version is starved of GIL under donor commit load (see
 * profile data — sweeps balloon from 10 ms to 215 ms under load),
 * because every CXL row read in `_arbitrate_lock_id` is a Python-level
 * attribute access that needs the GIL. This C program is a separate
 * OS process — no Python, no GIL, no starvation.
 *
 * Protocol: identical to the Python LockManager. We mmap the same
 * /dev/dax0.0 device, read the same header layout (see layout.py),
 * and run the same sweep:
 *
 *   for each lock_id in [0, num_locks):
 *     for each node_id in [0, max_nodes):
 *       CLFLUSH(row[lock_id][node_id])
 *     if any row.state == LOCKED: skip (current holder must release first)
 *     else: find WAITING row with smallest seq, set state = LOCKED, fence
 *
 * Build: gcc -O2 -march=native -pthread cxl_lock_manager.c -o cxl_lock_manager
 *
 * Usage:
 *   cxl_lock_manager --dev /dev/dax0.0 [--report-interval-s 10]
 *                    [--pool-size-override N] [--no-flush]
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#define HAS_X86 1
#else
#define HAS_X86 0
#endif

/* ---------- layout constants. MUST match layout.py. ---------------- */

#define HEADER_SIZE   4096u
#define LAYOUT_VERSION 1u
#define MAGIC_LE      0x4C4D43584C504F4FULL  /* "LMCXLPOO" little-endian */
#define CACHELINE     64u
#define GEOM_HASH_SZ  16u

#define LOCK_STATE_IDLE    0u
#define LOCK_STATE_WAITING 1u
#define LOCK_STATE_LOCKED  2u

/* layout.py Header (packed). We only need the lock-relevant fields,
 * but we read the whole thing for validation. */
struct Header {
    uint64_t magic;
    uint32_t layout_version;
    uint32_t _pad0;
    uint64_t gen;
    uint8_t  geom_hash[GEOM_HASH_SZ];
    uint64_t region_size;
    uint32_t region_count;
    uint32_t index_slot_count;
    uint32_t num_locks;
    uint32_t max_nodes;
    uint64_t off_global_locks;
    uint64_t off_region_bitmap;
    uint64_t off_region_descs;
    uint64_t off_index;
    uint64_t off_regions;
    uint64_t pool_size;
    uint32_t search_hint;
    uint32_t _pad1;
} __attribute__((packed));

/* One cell of global_lock[NUM_LOCKS][MAX_NODES]. 64 B = one cacheline. */
struct LockSlot {
    uint32_t state;
    uint32_t seq;
    uint8_t  _pad[CACHELINE - 8];
} __attribute__((packed));

_Static_assert(sizeof(struct LockSlot) == CACHELINE,
               "LockSlot must be 64 bytes");

/* ---------- fence primitives --------------------------------------- */

static int g_use_flush = 1;

static inline void flush_line(void *addr) {
#if HAS_X86
    if (g_use_flush) _mm_clflush(addr);
#else
    (void)addr;
#endif
}

static inline void mfence(void) {
#if HAS_X86
    if (g_use_flush) _mm_mfence();
#endif
}

/* ---------- signal handling ---------------------------------------- */

static volatile sig_atomic_t g_stop = 0;

static void on_signal(int sig) {
    (void)sig;
    g_stop = 1;
}

/* ---------- arbitration -------------------------------------------- */

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Returns 1 if a grant was issued, 0 otherwise. Mirrors
 * lock_manager.py:_arbitrate_lock_id. */
static int arbitrate_lock_id(struct LockSlot *row, uint32_t max_nodes) {
    /* Flush the whole row before reading so we observe peer writes. */
    for (uint32_t n = 0; n < max_nodes; ++n) {
        flush_line(&row[n]);
    }

    int best_idx = -1;
    uint32_t best_seq = 0;
    int have_best = 0;
    for (uint32_t n = 0; n < max_nodes; ++n) {
        uint32_t state = row[n].state;
        if (state == LOCK_STATE_LOCKED) {
            return 0;  /* current holder must release before we grant next */
        }
        if (state == LOCK_STATE_WAITING) {
            uint32_t seq = row[n].seq;
            if (!have_best || seq < best_seq) {
                best_seq = seq;
                best_idx = (int)n;
                have_best = 1;
            }
        }
    }

    if (best_idx < 0) return 0;

    /* Grant the winner. */
    row[best_idx].state = LOCK_STATE_LOCKED;
    flush_line(&row[best_idx]);
    mfence();
    return 1;
}

/* ---------- DAX device size discovery ------------------------------ */

/* Return the basename of a path, e.g. "/dev/dax0.0" -> "dax0.0".
 * Result points into the input string. */
static const char *path_basename(const char *path) {
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

/* Read /sys/bus/dax/devices/<name>/size. Returns 0 on failure. */
static uint64_t sysfs_dax_size(const char *dev_path) {
    char buf[256];
    int n = snprintf(buf, sizeof(buf),
                     "/sys/bus/dax/devices/%s/size",
                     path_basename(dev_path));
    if (n <= 0 || (size_t)n >= sizeof(buf)) return 0;
    FILE *f = fopen(buf, "r");
    if (!f) return 0;
    char line[64];
    line[0] = '\0';
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return 0;
    }
    fclose(f);
    char *end = NULL;
    unsigned long long v = strtoull(line, &end, 10);
    if (end == line) return 0;
    return (uint64_t)v;
}

/* Run `daxctl list -d <name> -j` and parse the first integer that
 * follows a `"size":` token. daxctl emits well-formed JSON so a
 * substring search is safe enough here.  Returns 0 on failure. */
static uint64_t daxctl_dax_size(const char *dev_path) {
    char cmd[256];
    const char *name = path_basename(dev_path);
    int n = snprintf(cmd, sizeof(cmd),
                     "daxctl list -d %s -j 2>/dev/null", name);
    if (n <= 0 || (size_t)n >= sizeof(cmd)) return 0;
    FILE *p = popen(cmd, "r");
    if (!p) return 0;

    char buf[8192];
    size_t total = 0;
    size_t got;
    while ((got = fread(buf + total, 1, sizeof(buf) - 1 - total, p)) > 0) {
        total += got;
        if (total >= sizeof(buf) - 1) break;
    }
    buf[total] = '\0';
    int rc = pclose(p);
    if (rc != 0 || total == 0) return 0;

    /* Look for "size":<digits>. */
    const char *key = "\"size\":";
    char *hit = strstr(buf, key);
    if (!hit) return 0;
    hit += strlen(key);
    while (*hit == ' ' || *hit == '\t') ++hit;
    char *end = NULL;
    unsigned long long v = strtoull(hit, &end, 10);
    if (end == hit) return 0;
    return (uint64_t)v;
}

/* Combined probe. */
static uint64_t dax_device_size(const char *dev_path) {
    uint64_t s = sysfs_dax_size(dev_path);
    if (s != 0) return s;
    return daxctl_dax_size(dev_path);
}

/* ---------- main loop ---------------------------------------------- */

struct Args {
    const char *dev_path;
    double report_interval_s;
    uint64_t pool_size_override;
    int use_flush;
};

static int parse_args(int argc, char **argv, struct Args *out) {
    out->dev_path = NULL;
    out->report_interval_s = 10.0;
    out->pool_size_override = 0;
    out->use_flush = 1;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (strcmp(a, "--dev") == 0 && i + 1 < argc) {
            out->dev_path = argv[++i];
        } else if (strcmp(a, "--report-interval-s") == 0 && i + 1 < argc) {
            out->report_interval_s = atof(argv[++i]);
        } else if (strcmp(a, "--pool-size-override") == 0 && i + 1 < argc) {
            out->pool_size_override = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(a, "--no-flush") == 0) {
            out->use_flush = 0;
        } else if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            fprintf(stderr,
                "Usage: %s --dev /dev/dax0.0 [--report-interval-s 10] "
                "[--pool-size-override N] [--no-flush]\n",
                argv[0]);
            return -1;
        } else {
            fprintf(stderr, "unknown arg: %s\n", a);
            return -1;
        }
    }
    if (!out->dev_path) {
        fprintf(stderr, "missing required --dev\n");
        return -1;
    }
    return 0;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;
    if (x < y) return -1;
    if (x > y) return 1;
    return 0;
}

int main(int argc, char **argv) {
    struct Args args;
    if (parse_args(argc, argv, &args) != 0) return 2;
    g_use_flush = args.use_flush;

    struct sigaction sa = {0};
    sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    int fd = open(args.dev_path, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "open(%s): %s\n", args.dev_path, strerror(errno));
        return 1;
    }

    /* Discover the full device size. DAX char devices require mmap
     * with alignment matching the device's mapping alignment (typically
     * 2 MiB) and fstat reports st_size==0 for them — so we can't mmap
     * just the 4 KiB header first. We must mmap the whole pool in one
     * call, matching what the Python bootstrap does.
     *
     * Probe order, mirroring bootstrap.py:_open_pool:
     *   1. fstat — works for regular files (tests use tmpfiles).
     *   2. /sys/bus/dax/devices/<name>/size — sysfs, no root needed.
     *   3. daxctl list -d <name> -j — JSON fallback.
     *
     * pool_size_override, if given, caps the discovered size (matches
     * Python's behavior where pool_size_override is a cap on the
     * mapping size).
     */
    uint64_t pool_size = 0;
    {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > 0) {
            pool_size = (uint64_t)st.st_size;
        } else if (fstat(fd, &st) == 0 && S_ISCHR(st.st_mode)) {
            pool_size = dax_device_size(args.dev_path);
        }
    }
    if (pool_size == 0) {
        fprintf(stderr,
            "could not determine size of %s; sysfs and daxctl probes "
            "failed. If this is a DAX device, verify it is enabled "
            "(`daxctl list`) and that /sys/bus/dax/devices/<name>/size "
            "is readable.\n",
            args.dev_path);
        close(fd);
        return 1;
    }
    if (args.pool_size_override != 0 && args.pool_size_override < pool_size) {
        pool_size = args.pool_size_override;
    }

    void *pool = mmap(NULL, pool_size, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, 0);
    if (pool == MAP_FAILED) {
        fprintf(stderr, "mmap(pool, %llu): %s\n",
                (unsigned long long)pool_size, strerror(errno));
        close(fd);
        return 1;
    }

    /* Read + validate the header from the pool mapping. */
    flush_line(pool);
    mfence();
    struct Header hdr;
    memcpy(&hdr, pool, sizeof(hdr));

    if (hdr.magic != MAGIC_LE) {
        fprintf(stderr, "bad magic 0x%016llx, expected 0x%016llx — "
                "pool not initialized yet?\n",
                (unsigned long long)hdr.magic,
                (unsigned long long)MAGIC_LE);
        munmap(pool, pool_size);
        close(fd);
        return 1;
    }
    if (hdr.layout_version != LAYOUT_VERSION) {
        fprintf(stderr, "layout version mismatch: header=%u expected=%u\n",
                hdr.layout_version, LAYOUT_VERSION);
        munmap(pool, pool_size);
        close(fd);
        return 1;
    }

    struct LockSlot *lock_array =
        (struct LockSlot *)((uint8_t *)pool + hdr.off_global_locks);
    uint32_t num_locks = hdr.num_locks;
    uint32_t max_nodes = hdr.max_nodes;

    fprintf(stderr,
        "cxl_lock_manager: dev=%s pool_size=%llu num_locks=%u max_nodes=%u "
        "off_global_locks=%llu use_flush=%d gen=%llu\n",
        args.dev_path,
        (unsigned long long)pool_size, num_locks, max_nodes,
        (unsigned long long)hdr.off_global_locks, g_use_flush,
        (unsigned long long)hdr.gen);
    fflush(stderr);

    /* Sweep loop with periodic reporting. */
    const size_t MAX_SAMPLES = 32768;
    uint64_t *samples = (uint64_t *)malloc(MAX_SAMPLES * sizeof(uint64_t));
    if (!samples) {
        fprintf(stderr, "out of memory for samples buffer\n");
        return 1;
    }
    size_t n_samples = 0;
    uint64_t sweep_ns_sum = 0;
    uint64_t sweep_ns_max = 0;
    uint64_t grants_in_window = 0;
    uint64_t last_report_ns = now_ns();
    uint64_t report_period_ns =
        (uint64_t)(args.report_interval_s * 1e9);

    while (!g_stop) {
        uint64_t t0 = now_ns();
        uint64_t grants = 0;
        for (uint32_t lock_id = 0; lock_id < num_locks; ++lock_id) {
            struct LockSlot *row = &lock_array[(size_t)lock_id * max_nodes];
            grants += (uint64_t)arbitrate_lock_id(row, max_nodes);
        }
        uint64_t dt = now_ns() - t0;
        sweep_ns_sum += dt;
        if (dt > sweep_ns_max) sweep_ns_max = dt;
        if (n_samples < MAX_SAMPLES) samples[n_samples++] = dt;
        grants_in_window += grants;

        uint64_t now = now_ns();
        if (now - last_report_ns >= report_period_ns) {
            double window_s = (double)(now - last_report_ns) / 1e9;
            double mean_us =
                (n_samples ? (double)sweep_ns_sum / (double)n_samples : 0.0)
                / 1000.0;
            double max_us = (double)sweep_ns_max / 1000.0;
            qsort(samples, n_samples, sizeof(uint64_t), cmp_u64);
            double p50_us =
                n_samples ? (double)samples[n_samples / 2] / 1000.0 : 0.0;
            size_t p99_idx = n_samples
                ? (size_t)((double)n_samples * 0.99)
                : 0;
            if (p99_idx > 0) p99_idx -= 1;
            double p99_us =
                n_samples ? (double)samples[p99_idx] / 1000.0 : 0.0;
            double sweeps_per_s = (double)n_samples / window_s;
            double grants_per_s = (double)grants_in_window / window_s;
            fprintf(stderr,
                "cxl_lock_manager: sweeps=%zu sweeps/s=%.0f grants/s=%.1f "
                "sweep_us[mean=%.1f p50=%.1f p99=%.1f max=%.1f]\n",
                n_samples, sweeps_per_s, grants_per_s,
                mean_us, p50_us, p99_us, max_us);
            fflush(stderr);
            n_samples = 0;
            sweep_ns_sum = 0;
            sweep_ns_max = 0;
            grants_in_window = 0;
            last_report_ns = now;
        }
    }

    free(samples);
    munmap(pool, pool_size);
    close(fd);
    fprintf(stderr, "cxl_lock_manager: stopped (signal received)\n");
    return 0;
}
