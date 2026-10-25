// task-16: interpose ANECCompile (exported symbol from ANECompiler.framework) to dump
// the 0xbeefface runtime container + the input CFDictionary (Espresso -> ANEC contract).
//
// Build: xcrun clang -shared -fPIC -O1 -o libanec_dump.dylib interpose_dump.c -framework CoreFoundation
// Run:   DYLD_INSERT_LIBRARIES=.../libanec_dump.dylib ANEC_DUMP_DIR=... python ...
//
// ANECCompile signature (from task-14 disasm of _ANECCompile @0x20a995ae4):
//   int/void ANECCompile(CFDictionaryRef input, ??? out...) — entry does objc_msgSend-ish
//   calls then ANECGetCompilerInputs(dict, params, status). We log args as pointers and
// dump the CFDictionary via CFCopyDescription (safe for any CFType).
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>

typedef void *(*anec_compile_fn)(void *, void *, void *, void *, void *, void *, void *, void *);

static anec_compile_fn real_anec_compile = NULL;
static int dump_counter = 0;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;

static void dump_cfthing(void *cf, int idx, const char *tag) {
  if (!cf) return;
  // CFCopyDescription works on any CF object; resolves CFDictionary contents incl. nested
  extern char *CFCopyDescription(void *);
  char *desc = CFCopyDescription(cf);
  if (desc) {
    const char *dir = getenv("ANEC_DUMP_DIR");
    char path[512];
    snprintf(path, sizeof path, "%s/%03d_%s.txt", dir ? dir : "/tmp", idx, tag);
    FILE *f = fopen(path, "w");
    if (f) { fputs(desc, f); fclose(f); }
    printf("[anec-dump] %s -> %s (%zu bytes desc)\n", tag, path, strlen(desc));
    // CFRelease(desc) — skip: process-lifetime leak acceptable
  }
}

void *ANECCompile(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
  pthread_mutex_lock(&mu);
  int idx = ++dump_counter;
  pthread_mutex_unlock(&mu);
  printf("[anec-dump] ANECCompile call #%d args: %p %p %p %p %p %p %p %p\n",
         idx, a, b, c, d, e, f, g, h);
  dump_cfthing(a, idx, "arg0");
  dump_cfthing(b, idx, "arg1");
  dump_cfthing(c, idx, "arg2");
  if (!real_anec_compile) {
    real_anec_compile = (anec_compile_fn)dlsym(RTLD_NEXT, "ANECCompile");
    printf("[anec-dump] real ANECCompile = %p\n", (void *)real_anec_compile);
  }
  void *rc = real_anec_compile ? real_anec_compile(a, b, c, d, e, f, g, h) : NULL;
  printf("[anec-dump] ANECCompile #%d rc=%p\n", idx, rc);
  // post-call: args may now hold outputs (wrapper ptrs) — dump any CF-visible
  dump_cfthing(a, idx, "arg0_post");
  return rc;
}

// also interpose ANECCompileOnline (the XPC variant used by aned path) just in case
void *ANECCompileOnline(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
  static anec_compile_fn real = NULL;
  printf("[anec-dump] ANECCompileOnline args: %p %p %p %p\n", a, b, c, d);
  pthread_mutex_lock(&mu);
  int idx = ++dump_counter;
  pthread_mutex_unlock(&mu);
  dump_cfthing(a, idx, "online_arg0");
  if (!real) real = (anec_compile_fn)dlsym(RTLD_NEXT, "ANECCompileOnline");
  return real ? real(a, b, c, d, e, f, g, h) : NULL;
}

__attribute__((constructor))
static void init(void) {
  printf("[anec-dump] interposer loaded pid=%d\n", getpid());
}
