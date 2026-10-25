// surf_hook.c — in-process IOSurface receipt interception (task-17 channel b).
//
// Mechanism: rebinding (fishhook-style) of IOSurfaceLookup /
// IOSurfaceLookupFromMachPort / IOSurfaceGetBytePointer in OUR process.
// Espresso (dlopen'd by aneforge) receives the runtime container (model.hwx,
// 0xbeefface) as an IOSurface transferred over the ANEVirtualClient IPC —
// every receipt funnels through these public CoreFoundation calls. Rebidding
// happens on our own process's symbol-binding tables (no DYLD_INSERT, no
// AMFI involvement — legal memory operations on self).
//
// Modes via env:
//   ANE_SURF_DUMP=<dir>   dump every received surface's bytes to <dir>/surf_N.bin
//   ANE_SURF_MIN=<bytes>  skip surfaces smaller than this (default 1024)
//   ANE_SURF_PATCH=<file> patch spec applied in-place before returning the
//                         byte pointer (surgery channel): "off:hex[,off:hex...]"
//                         applied at each 0xbeefface magic found in the surface
//
// Build: cc -dynamiclib -o libsurf_hook.dylib surf_hook.c -framework IOSurface -framework CoreFoundation
// Use:   ctypes.CDLL("./libsurf_hook.dylib")  (constructor installs rebind)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <mach-o/nlist.h>
#include <sys/stat.h>
#include <IOSurface/IOSurface.h>
#include <CoreFoundation/CoreFoundation.h>

// IOSurfaceGetBytePointer is C++-linkage in the header; resolve dynamically:
static void *(*g_get_byte_ptr)(IOSurfaceRef) = NULL;
static void init_byteptr(void) {
    if (!g_get_byte_ptr)
        g_get_byte_ptr = (void *(*)(IOSurfaceRef))dlsym(RTLD_DEFAULT, "IOSurfaceGetBytePointer");
}
#define SURF_BYTES(s) (init_byteptr(), g_get_byte_ptr ? g_get_byte_ptr(s) : NULL)

#define LOG(...) do { if (g_verbose) fprintf(stderr, "[surfhook] " __VA_ARGS__); } while (0)
static int g_verbose = 0;
static char g_dumpdir[512];
static size_t g_min_size = 1024;
static uint32_t g_counter = 0;

typedef IOSurfaceRef (*lookup_fn)(uint32_t csid);
typedef IOSurfaceRef (*lookup_mp_fn)(mach_port_t port);
typedef void *(*byteptr_fn)(IOSurfaceRef s);

static lookup_fn real_lookup = NULL;
static lookup_mp_fn real_lookup_mp = NULL;
static byteptr_fn real_byteptr = NULL;

// ---- patch spec -------------------------------------------------------------
static struct { size_t off; uint8_t bytes[8]; size_t n; } g_patches[16];
static int g_npatches = 0;

static void parse_patch_spec(void) {
    const char *spec = getenv("ANE_SURF_PATCH");
    if (!spec) return;
    char *tmp = strdup(spec), *save = NULL;
    for (char *tok = strtok_r(tmp, ",", &save); tok && g_npatches < 16;
         tok = strtok_r(NULL, ",", &save)) {
        char *colon = strchr(tok, ':');
        if (!colon) continue;
        *colon = 0;
        g_patches[g_npatches].off = strtoull(tok, NULL, 0);
        const char *hex = colon + 1;
        size_t n = 0;
        for (size_t i = 0; hex[i] && hex[i+1] && n < 8; i += 2) {
            unsigned v; if (sscanf(hex + i, "%2x", &v) == 1)
                g_patches[g_npatches].bytes[n++] = (uint8_t)v;
        }
        g_patches[g_npatches].n = n;
        g_npatches++;
    }
    free(tmp);
}

static void dump_and_patch(const char *tag, void *bytes, size_t len) {
    if (!bytes || len < g_min_size) return;
    // container scan + optional surgery
    for (size_t i = 0; i + 4 <= len; i += 4) {
        uint32_t m; memcpy(&m, (char *)bytes + i, 4);
        if (m != 0xBEEFFACE) continue;
        LOG("%s: 0xbeefface @ +%#zx (len=%#zx)\n", tag, i, len);
        for (int p = 0; p < g_npatches; p++) {
            size_t at = i + g_patches[p].off;
            if (at + g_patches[p].n <= len) {
                memcpy((char *)bytes + at, g_patches[p].bytes, g_patches[p].n);
                LOG("  patched %#zx+%zu bytes\n", g_patches[p].off, g_patches[p].n);
            }
        }
    }
    if (!g_dumpdir[0]) return;
    char path[600];
    snprintf(path, sizeof(path), "%s/surf_%u_%s_%zu.bin", g_dumpdir,
             __sync_fetch_and_add(&g_counter, 1), tag, len);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return;
    size_t off = 0;
    while (off < len) {
        ssize_t w = write(fd, (char *)bytes + off, len - off);
        if (w <= 0) break;
        off += (size_t)w;
    }
    close(fd);
    LOG("dumped %zu bytes -> %s\n", len, path);
}

// ---- trampolines ------------------------------------------------------------
static IOSurfaceRef my_lookup(uint32_t csid) {
    IOSurfaceRef s = real_lookup(csid);
    if (s) {
        size_t len = IOSurfaceGetAllocSize(s);
        void *p = SURF_BYTES(s);
        dump_and_patch("lookup", p, len);
    }
    return s;
}

static IOSurfaceRef my_lookup_mp(mach_port_t port) {
    IOSurfaceRef s = real_lookup_mp(port);
    if (s) {
        size_t len = IOSurfaceGetAllocSize(s);
        void *p = SURF_BYTES(s);
        dump_and_patch("lookupmp", p, len);
    }
    return s;
}

// ---- fishhook-style rebinding ----------------------------------------------
static void rebind_symbols(void) {
    uint32_t count = _dyld_image_count();
    for (uint32_t img = 0; img < count; img++) {
        const struct mach_header *hdr = _dyld_get_image_header(img);
        intptr_t slide = _dyld_get_image_vmaddr_slide(img);
        if (!hdr || hdr->magic != MH_MAGIC_64) continue;
        const struct load_command *lc = (const void *)((char *)hdr + sizeof(*hdr));
        const struct dysymtab_command *dysym = NULL;
        const struct symtab_command *symtab = NULL;
        for (uint32_t c = 0; c < hdr->ncmds; c++, lc = (const void *)((char *)lc + lc->cmdsize)) {
            if (lc->cmd == LC_SYMTAB) symtab = (const void *)lc;
            if (lc->cmd == LC_DYSYMTAB) dysym = (const void *)lc;
        }
        if (!symtab || !dysym) continue;
        const struct nlist_64 *syms = (const void *)((char *)hdr + symtab->symoff);
        const uint32_t *itab = (const uint32_t *)((char *)hdr + dysym->indirectsymoff);
        lc = (const void *)((char *)hdr + sizeof(*hdr));
        for (uint32_t c = 0; c < hdr->ncmds; c++, lc = (const void *)((char *)lc + lc->cmdsize)) {
            if (lc->cmd != LC_SEGMENT_64) continue;
            const struct segment_command_64 *seg = (const void *)lc;
            if (strcmp(seg->segname, "__DATA")) continue;  // writable only: __DATA_CONST uses chained fixups on macOS 26
            const struct section_64 *sec = (const void *)((char *)seg + sizeof(*seg));
            for (uint32_t s = 0; s < seg->nsects; s++, sec++) {
                if (strcmp(sec->sectname, "__la_symbol_ptr")) continue;
                uint32_t idx0 = sec->reserved1;
                void **ptrs = (void **)(sec->addr + slide);
                for (uint32_t k = 0; k < sec->size / sizeof(void *); k++) {
                    uint32_t si = itab[idx0 + k];
                    if (si == INDIRECT_SYMBOL_ABS || si == INDIRECT_SYMBOL_LOCAL) continue;
                    const char *name = (char *)hdr + symtab->stroff + syms[si].n_un.n_strx;
                    if (!strcmp(name, "_IOSurfaceLookup") && !real_lookup) {
                        real_lookup = (lookup_fn)ptrs[k];
                        ptrs[k] = (void *)my_lookup;
                        LOG("rebound IOSurfaceLookup in image %u\n", img);
                    } else if (!strcmp(name, "_IOSurfaceLookupFromMachPort") && !real_lookup_mp) {
                        real_lookup_mp = (lookup_mp_fn)ptrs[k];
                        ptrs[k] = (void *)my_lookup_mp;
                        LOG("rebound IOSurfaceLookupFromMachPort in image %u\n", img);
                    }
                }
            }
        }
    }
    if (!real_lookup) real_lookup = (lookup_fn)dlsym(RTLD_NEXT, "IOSurfaceLookup");
    if (!real_lookup_mp) real_lookup_mp = (lookup_mp_fn)dlsym(RTLD_NEXT, "IOSurfaceLookupFromMachPort");
    LOG("rebind done: lookup=%p lookupmp=%p\n", (void *)real_lookup, (void *)real_lookup_mp);
}

__attribute__((constructor))
static void init(void) {
    g_verbose = getenv("ANE_SURF_VERBOSE") != NULL;
    const char *d = getenv("ANE_SURF_DUMP");
    if (d) { snprintf(g_dumpdir, sizeof(g_dumpdir), "%s", d); mkdir(d, 0755); }
    const char *m = getenv("ANE_SURF_MIN");
    if (m) g_min_size = strtoull(m, NULL, 0);
    parse_patch_spec();
    rebind_symbols();
    LOG("init: dump=%s min=%zu patches=%d\n", g_dumpdir, g_min_size, g_npatches);
}
