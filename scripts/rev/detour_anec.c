// task-16: mach_vm trampoline detour on ANECCompile (runtime addr via dlsym) —
// the dyld-cache two-level namespace defeats DYLD_INSERT preempt, so we patch
// the prologue with an absolute jump to our detour, log args + CFDict, then run
// the displaced instructions and jump back.
//
// Build into the SAME process at runtime (called from Python via ctypes):
//   trampoline_setup() -> installs detour, returns 0 on success
//
// Detour behavior:
//   - logs the 8 arg registers
//   - CFCopyDescription(arg0/arg1) -> files in $ANEC_DUMP_DIR
//   - calls the original (via trampoline that re-executes displaced insns)
//   - scans returned pointers for Mach-O magic 0xbeefface / 0xfeedface in pointees
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>

typedef void *(*fn8)(void *, void *, void *, void *, void *, void *, void *, void *);

static void *g_target = NULL;      // ANECCompile runtime addr
static void *g_trampoline = NULL;  // executable page: displaced insns + jump back
static unsigned char g_displaced[16];
static int g_count = 0;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;

extern char *CFCopyDescription(void *);

static void dump_cf(void *cf, int idx, const char *tag) {
  if (!cf) return;
  char *desc = CFCopyDescription(cf);
  if (!desc) return;
  const char *dir = getenv("ANEC_DUMP_DIR");
  char path[512];
  snprintf(path, sizeof path, "%s/%03d_%s.txt", dir ? dir : "/tmp", idx, tag);
  FILE *f = fopen(path, "w");
  if (f) { size_t n = strlen(desc); fwrite(desc, 1, n, f); fclose(f); }
  printf("[detour] %s -> %s\n", tag, path);
}

// Scan candidate pointers returned by ANECCompile for Mach-O magics.
static void scan_container(void *p, int idx, const char *tag) {
  if (!p) return;
  for (int lvl = 0; lvl < 3; lvl++) {
    void **v = (void **)p;
    for (int i = 0; i < 16; i++) {
      void *q = v[i];
      if (!q) continue;
      unsigned int m = *(unsigned int *)((char *)q);
      // MH_MAGIC / MH_MAGIC_64 / fat / 0xbeefface candidates
      if (m == 0xfeedface || m == 0xfeedfacf || m == 0xcafebabe || m == 0xbeefface || m == 0xfacefeed) {
        const char *dir = getenv("ANEC_DUMP_DIR");
        char path[512];
        snprintf(path, sizeof path, "%s/%03d_%s_MachO_%x.bin", dir ? dir : "/tmp", idx, tag, m);
        // size unknown; cap at 4MB
        FILE *f = fopen(path, "wb");
        if (f) { fwrite(q, 1, 4 << 20, f); fclose(f); }
        printf("[detour] MachO magic %x at lvl%d slot%d -> %s\n", m, lvl, i, path);
      }
    }
    // follow first non-null pointer for next level
    if (!v[0]) break;
    p = v[0];
  }
}

static void *detour(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
  pthread_mutex_lock(&mu);
  int idx = ++g_count;
  pthread_mutex_unlock(&mu);
  printf("[detour] ANECCompile #%d: %p %p %p %p %p %p %p %p\n", idx, a, b, c, d, e, f, g, h);
  dump_cf(a, idx, "arg0");
  dump_cf(b, idx, "arg1");
  dump_cf(c, idx, "arg2");
  void *rc = ((fn8)g_trampoline)(a, b, c, d, e, f, g, h);
  printf("[detour] ANECCompile #%d rc=%p\n", idx, rc);
  dump_cf(a, idx, "arg0_post");
  dump_cf(rc, idx, "rc");
  scan_container(rc, idx, "rc");
  return rc;
}

static void make_rw(mach_vm_address_t addr, mach_vm_size_t size) {
  mach_vm_address_t page = addr & ~0x3fffULL;
  mach_vm_size_t sz = size + (addr - page);
  mach_vm_protect(mach_task_self(), page, sz, 0, VM_PROT_READ | VM_PROT_WRITE | VM_PROT_COPY);
}

static void make_rx(mach_vm_address_t addr, mach_vm_size_t size) {
  mach_vm_address_t page = addr & ~0x3fffULL;
  mach_vm_size_t sz = size + (addr - page);
  mach_vm_protect(mach_task_self(), page, sz, 0, VM_PROT_READ | VM_PROT_EXECUTE);
}

// write abs jump: 16 bytes: LDR X16, #8; BR X16; <8-byte addr>
static void write_abs_jump(void *dst, void *target) {
  unsigned char code[16];
  code[0] = 0x50; code[1] = 0x00; code[2] = 0x00; code[3] = 0x58;  // ldr x16, [pc, #8]
  code[4] = 0x00; code[5] = 0x02; code[6] = 0x1f; code[7] = 0xd6;  // br x16
  memcpy(code + 8, &target, 8);
  memcpy(dst, code, 16);
}

static void clear_cache(void *start, size_t len) {
  __builtin___clear_cache((char *)start, (char *)start + len);
}

int trampoline_setup(void) {
  void *h = dlopen("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler", RTLD_LAZY);
  if (!h) { printf("[detour] dlopen ANECompiler failed\n"); return -1; }
  void *sym = dlsym(h, "ANECCompile");
  if (!sym) { printf("[detour] dlsym failed\n"); return -2; }
  g_target = sym;
  printf("[detour] ANECCompile @ %p\n", g_target);

  // allocate trampoline page RWX
  mach_vm_address_t tramp = 0;
  kern_return_t kr = mach_vm_allocate(mach_task_self(), &tramp, 0x4000,
                                      VM_FLAGS_ANYWHERE);
  if (kr) { printf("[detour] alloc failed %d\n", kr); return -3; }
  mach_vm_protect(mach_task_self(), tramp, 0x4000, 0, VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE);
  g_trampoline = (void *)tramp;

  // ANECCompile prologue (from dsc extract):
  //   20a995ae4: d503237f  pacibsp
  //   20a995ae8: a9bf7bfd  stp x29, x30, [sp,#-0x10]!
  //   ... (task-14 dump: next is stp pair + sub sp)
  // Displace first 16 bytes -> trampoline: copy 16B + abs jump back to target+16.
  memcpy(g_displaced, g_target, 16);
  memcpy(g_trampoline, g_displaced, 16);
  write_abs_jump((char *)g_trampoline + 16, (char *)g_target + 16);
  clear_cache(g_trampoline, 32);

  // patch target prologue -> detour
  make_rw((mach_vm_address_t)g_target, 16);
  write_abs_jump(g_target, detour);
  clear_cache(g_target, 16);
  make_rx((mach_vm_address_t)g_target, 16);
  printf("[detour] trampoline installed @ %p -> detour %p\n", g_trampoline, (void *)detour);
  return 0;
}

int trampoline_count(void) { return g_count; }
