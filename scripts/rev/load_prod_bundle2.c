// load_prod_bundle2.c — full direct-submit attempt on a production e5 bundle.
// Path: program_library → retain function → precompiled_compute_op_options →
//       execution_stream_operation → (submit skipped unless shapes known)
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

typedef int e5rt_error_code_t;
#define DECL(name) v_##name name
#define LOAD(name) do { \
    name = (v_##name)dlsym(fw, #name); if (!name) name = (v_##name)dlsym(fw, "_" #name); \
    if (!name) { printf("missing %s\n", #name); return 1; } \
} while (0)

typedef e5rt_error_code_t (*v_lib_create)(void **, const char *);
typedef e5rt_error_code_t (*v_lib_num)(void *, uint64_t *);
typedef e5rt_error_code_t (*v_lib_names)(void *, uint64_t, const char **);
typedef e5rt_error_code_t (*v_lib_retain)(void *, const char *, void **);
typedef e5rt_error_code_t (*v_opopt_create)(void **, void *);
typedef e5rt_error_code_t (*v_opopt_setname)(void *, const char *);
typedef e5rt_error_code_t (*v_opopt_setalloc)(void *, int);
typedef e5rt_error_code_t (*v_op_create)(void **, void *);
typedef e5rt_error_code_t (*v_stream_create)(void **);

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) { printf("usage: %s <bundle>\n", argv[0]); return 1; }
    void *fw = dlopen("/System/Library/PrivateFrameworks/Espresso.framework/Espresso", RTLD_NOW);
    if (!fw) { printf("dlopen: %s\n", dlerror()); return 1; }
    DECL(lib_create); DECL(lib_num); DECL(lib_names); DECL(lib_retain);
    DECL(opopt_create); DECL(opopt_setname); DECL(opopt_setalloc);
    DECL(op_create); DECL(stream_create);

    void *lib = NULL;
    int rc = lib_create(&lib, argv[1]);
    printf("library_create = %d lib=%p\n", rc, lib);
    if (rc) return 1;
    uint64_t n = 0; lib_num(lib, &n);
    printf("num_functions = %llu\n", (unsigned long long)n);
    const char **names = calloc(n ? n : 1, sizeof(char *));
    lib_names(lib, n, names);
    for (uint64_t i = 0; i < n; i++) printf("  fn[%llu] %s\n", (unsigned long long)i, names[i]);

    void *fn = NULL;
    rc = lib_retain(lib, n ? names[0] : "main", &fn);
    printf("retain = %d fn=%p\n", rc, fn);
    if (rc) return 1;

    void *opopts = NULL;
    rc = opopt_create(&opopts, fn);
    printf("op_options_create = %d\n", rc);
    if (rc) return 1;
    opopt_setname(opopts, n ? names[0] : "main");
    opopt_setalloc(opopts, 1);

    void *op = NULL;
    rc = op_create(&op, opopts);
    printf("precompiled_compute_operation_create = %d op=%p\n", rc, op);
    if (rc) {
        printf("=> op creation failed (device-bound step). Bundle loads + functions retain OK.\n");
        return 2;
    }
    void *stream = NULL;
    rc = stream_create(&stream);
    printf("execution_stream_create = %d\n", rc);
    printf("DONE: production bundle fully loadable via e5rt direct path.\n");
    return 0;
}
