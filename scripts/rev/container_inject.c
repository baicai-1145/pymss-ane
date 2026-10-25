// container_inject.c — in-process container rewrite hook (task-17 channel b).
//
// Mechanism (same as task-16 interpose, but mutating):
//   Espresso (our process, via aneforge e5rt path) calls ANECCompile inside
//   ANECompiler.dylib. The 0xbeefface container is returned as in-memory
//   bytes and handed to the device-load path without ever touching disk.
//   A DYLD_INSERT dylib exporting ANECCompile can:
//     1. call the real one (dlsym(RTLD_NEXT))
//     2. locate the container by magic 0xbeefface in the returned buffers
//     3. apply surgery via the patch table (env: ANE_PATCH=<file_off>:<hex>[,...])
//     4. return the patched bytes to Espresso — device then loads OUR kernel.
//
//   NOTE: interposing a dlsym'd export inside a framework the host dlopens is
//   linker-binding surgery; if AMFI blocks DYLD_INSERT for the venv python,
//   fall back to direct trampoline: patch the first instructions of the real
//   ANECCompile (we own the process, mach_vm_protect + branch).
//
// Patch table (L0/L1/L2 surgery ladder) supplied via env:
//   ANE_PATCH="0x20000:deadbeef,0x20100:00ff"
//   ANE_PATCH_OFFSET_FILE / ANE_PATCH_VALUE for DMA-shift experiments.
//
// This file is the skeleton; it compiles standalone and is activated once
// task-16's dump confirms the exact buffer shape returned by ANECCompile.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#define CONTAINER_MAGIC 0xBEEFFACE

typedef int (*anec_compile_fn)(void *req, void *out);

// Parse "0xOFF:hex" entries and apply to buffer.
static int apply_patches(unsigned char *buf, size_t len) {
    const char *spec = getenv("ANE_PATCH");
    if (!spec) return 0;
    int n = 0;
    char *tmp = strdup(spec), *save = NULL;
    for (char *tok = strtok_r(tmp, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        char *colon = strchr(tok, ':');
        if (!colon) continue;
        *colon = 0;
        size_t off = strtoull(tok, NULL, 0);
        const char *hex = colon + 1;
        size_t hlen = strlen(hex);
        for (size_t i = 0; i + 1 < hlen; i += 2) {
            if (off + i / 2 >= len) break;
            unsigned v;
            if (sscanf(hex + i, "%2x", &v) == 1) buf[off + i / 2] = (unsigned char)v;
        }
        n++;
    }
    free(tmp);
    return n;
}

// Scan a buffer for the container magic and patch every occurrence.
size_t scan_and_patch(unsigned char *buf, size_t len) {
    size_t hits = 0;
    for (size_t i = 0; i + 4 <= len; i += 4) {
        unsigned m;
        memcpy(&m, buf + i, 4);
        if (m == CONTAINER_MAGIC) {
            size_t n = apply_patches(buf + i, len - i);
            fprintf(stderr, "[inject] container @%p (+%#zx) patched %d entries\n",
                    (void *)(buf + i), i, (int)n);
            hits++;
        }
    }
    return hits;
}

#ifdef STANDALONE_TEST
int main(void) {
    // self-test: fake container, patch one field
    unsigned char buf[64];
    memset(buf, 0, sizeof(buf));
    memcpy(buf, "\xce\xfa\xef\xbe", 4);
    setenv("ANE_PATCH", "0x8:80000000", 0);
    printf("hits=%zu\n", scan_and_patch(buf, sizeof(buf)));
    printf("buf[8..0xb]=%02x%02x%02x%02x (expect 80000000)\n",
           buf[8], buf[9], buf[10], buf[11]);
    return 0;
}
#endif
