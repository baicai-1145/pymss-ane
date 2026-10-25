// task-16: interpose libxpc in OUR process — the ANECompilerService compile request +
// reply both transit through xpc_connection_send_message*_with_reply* here.
// We are inside the python process (ctypes dlopen), so a plain symbol preempt of
// libxpc functions via DYLD_INSERT is unnecessary: we can trampoline-patch
// xpc_connection_send_message_with_reply_sync (synchronous, easiest to wrap).
//
// Dump: every xpc reply dictionary containing data >= 4KB is written out and scanned
// for Mach-O magics. Plus log request dictionary descriptions (compile options contract!).
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>

typedef void *xpc_connection_t;
typedef void *xpc_object_t;

static void *g_target = NULL;
static void *g_trampoline = NULL;
static int g_count = 0;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;

extern void xpc_dictionary_apply(xpc_object_t, void *, void *);
extern const char *xpc_dictionary_get_string(xpc_object_t, const char *);
extern long xpc_dictionary_get_int64(xpc_object_t, const char *);
extern size_t xpc_dictionary_get_count(xpc_object_t);
extern const void *xpc_dictionary_get_data(xpc_object_t, const char *, size_t *);

static void maybe_dump_object(xpc_object_t obj, int idx, const char *tag) {
  if (!obj) return;
  const char *dir = getenv("ANEC_DUMP_DIR");
  // dump every dictionary with >=1 key for the contract record
  size_t n = xpc_dictionary_get_count(obj);
  char path[512];
  snprintf(path, sizeof path, "%s/%03d_%s.dict.bin", dir ? dir : "/tmp", idx, tag);
  // xpc_copy_description is not extern-C callable reliably; use xpc_dictionary_apply-less approach:
  // iterate known ANEC keys and log
  const char *keys[] = {"product", "output", "blob", "network", "program", "macho",
                        "status", "error", "compiledModel", "model", "bytes", "result", ""};
  for (int i = 0; keys[i][0]; i++) {
    const char *s = xpc_dictionary_get_string(obj, keys[i]);
    if (s) { printf("[xpc] %s.%s = \"%s\"\n", tag, keys[i], s); continue; }
    size_t dl = 0;
    const void *d = xpc_dictionary_get_data(obj, keys[i], &dl);
    if (d && dl >= 256) {
      snprintf(path, sizeof path, "%s/%03d_%s_%s_%zubin", dir ? dir : "/tmp", idx, tag, keys[i], dl);
      FILE *f = fopen(path, "wb");
      if (f) { fwrite(d, 1, dl > (8 << 20) ? (8 << 20) : dl, f); fclose(f); }
      unsigned int m = *(const unsigned int *)d;
      printf("[xpc] %s.%s = DATA %zu bytes magic=%08x -> %s\n", tag, keys[i], dl, m, path);
    }
  }
}

typedef xpc_object_t (*send_sync_fn)(xpc_connection_t, xpc_object_t);
static send_sync_fn real_send_sync = NULL;

static xpc_object_t detour_send_sync(xpc_connection_t c, xpc_object_t msg) {
  pthread_mutex_lock(&mu);
  int idx = ++g_count;
  pthread_mutex_unlock(&mu);
  maybe_dump_object(msg, idx, "req");
  xpc_object_t rc = real_send_sync ? real_send_sync(c, msg)
                                   : ((send_sync_fn)g_trampoline)(c, msg);
  maybe_dump_object(rc, idx, "rep");
  return rc;
}

static void make_rw(mach_vm_address_t addr, mach_vm_size_t size) {
  mach_vm_address_t page = addr & ~0x3fffULL;
  mach_vm_protect(mach_task_self(), page, size + (addr - page), 0,
                  VM_PROT_READ | VM_PROT_WRITE | VM_PROT_COPY);
}
static void make_rx(mach_vm_address_t addr, mach_vm_size_t size) {
  mach_vm_address_t page = addr & ~0x3fffULL;
  mach_vm_protect(mach_task_self(), page, size + (addr - page), 0,
                  VM_PROT_READ | VM_PROT_EXECUTE);
}
static void write_abs_jump(void *dst, void *target) {
  unsigned char code[16] = {0x50, 0x00, 0x00, 0x58, 0x00, 0x02, 0x1f, 0xd6};
  memcpy(code + 8, &target, 8);
  memcpy(dst, code, 16);
}

int xpc_interpose_setup(void) {
  void *h = dlopen("/usr/lib/system/libxpc.dylib", RTLD_LAZY);
  if (!h) { printf("[xpc] dlopen libxpc failed\n"); return -1; }
  void *sym = dlsym(h, "xpc_connection_send_message_with_reply_sync");
  if (!sym) { printf("[xpc] symbol not found\n"); return -2; }
  g_target = sym;
  printf("[xpc] send_message_with_reply_sync @ %p\n", g_target);

  mach_vm_address_t tramp = 0;
  if (mach_vm_allocate(mach_task_self(), &tramp, 0x4000, VM_FLAGS_ANYWHERE))
    return -3;
  mach_vm_protect(mach_task_self(), tramp, 0x4000, 0, VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE);
  g_trampoline = (void *)tramp;
  memcpy(g_trampoline, g_target, 16);
  write_abs_jump((char *)g_trampoline + 16, (char *)g_target + 16);
  __builtin___clear_cache((char *)g_trampoline, (char *)g_trampoline + 32);

  make_rw((mach_vm_address_t)g_target, 16);
  write_abs_jump(g_target, detour_send_sync);
  __builtin___clear_cache((char *)g_target, (char *)g_target + 16);
  make_rx((mach_vm_address_t)g_target, 16);
  printf("[xpc] trampoline installed\n");
  return 0;
}

int xpc_interpose_count(void) { return g_count; }
