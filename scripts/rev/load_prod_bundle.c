// load_prod_bundle.c — load a production H16G.bundle (VAD/keylm) via Espresso's
// e5rt program-library API and enumerate its functions. Runtime-layer submit path.
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

typedef int e5rt_error_code_t;
typedef e5rt_error_code_t (*lib_create_fn)(void **, const char *);
typedef e5rt_error_code_t (*lib_get_num_fn)(void *, uint64_t *);
typedef e5rt_error_code_t (*lib_get_names_fn)(void *, uint64_t, const char **);
typedef e5rt_error_code_t (*lib_retain_fn)(void *, const char *, void **);
typedef e5rt_error_code_t (*fn_load_fn)(void *);
typedef e5rt_error_code_t (*fn_release_fn)(void **);

#define SYM(var, name) \
    var = (var##_fn)dlsym(fw, name); if (!var) var = (var##_fn)dlsym(fw, "_" name)

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: %s <H16G.bundle path>\n", argv[0]); return 1; }
    void *fw = dlopen("/System/Library/PrivateFrameworks/Espresso.framework/Espresso", RTLD_NOW);
    if (!fw) { printf("dlopen Espresso: %s\n", dlerror()); return 1; }
    lib_create_fn lib_create; lib_get_num_fn lib_get_num; lib_get_names_fn lib_get_names;
    lib_retain_fn lib_retain; fn_load_fn fn_load; fn_release_fn fn_release;
    SYM(lib_create, "e5rt_program_library_create");
    SYM(lib_get_num, "e5rt_program_library_get_num_functions");
    SYM(lib_get_names, "e5rt_program_library_get_function_names");
    SYM(lib_retain, "e5rt_program_library_retain_program_function");
    SYM(fn_load, "e5rt_program_function_load_for_execution");
    SYM(fn_release, "e5rt_program_function_release");
    if (!lib_create || !lib_get_num || !lib_get_names || !lib_retain) {
        printf("missing symbols\n"); return 1;
    }
    void *lib = NULL;
    int rc = lib_create(&lib, argv[1]);
    printf("library_create(%s) = %d, lib=%p\n", argv[1], rc, lib);
    if (rc != 0 || !lib) return 1;
    uint64_t n = 0;
    lib_get_num(lib, &n);
    printf("num_functions = %llu\n", (unsigned long long)n);
    if (n && lib_get_names) {
        const char **names = calloc(n, sizeof(char *));
        if (lib_get_names(lib, n, names) == 0) {
            for (uint64_t i = 0; i < n; i++)
                printf("  fn[%llu] = %s\n", (unsigned long long)i, names[i] ? names[i] : "(null)");
        }
        if (n && names[0] && lib_retain) {
            void *fn = NULL;
            rc = lib_retain(lib, names[0], &fn);
            printf("retain(%s) = %d fn=%p\n", names[0], rc, fn);
            if (rc == 0 && fn && fn_load) {
                rc = fn_load(fn);
                printf("load_for_execution = %d\n", rc);
            }
            if (rc == 0 && fn && fn_release) {
                fn_release(&fn);
                printf("released\n");
            }
        }
    }
    return 0;
}
