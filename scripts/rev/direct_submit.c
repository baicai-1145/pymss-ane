#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
typedef int e5rt_error_code_t;
typedef e5rt_error_code_t (*f_lib_create)(void **, const char *);
typedef e5rt_error_code_t (*f_lib_num)(void *, uint64_t *);
typedef e5rt_error_code_t (*f_lib_names)(void *, uint64_t, const char **);
typedef e5rt_error_code_t (*f_lib_retain)(void *, const char *, void **);
typedef e5rt_error_code_t (*f_opopt_create)(void **, void *);
typedef e5rt_error_code_t (*f_opopt_name)(void *, const char *);
typedef e5rt_error_code_t (*f_opopt_alloc)(void *, int);
typedef e5rt_error_code_t (*f_op_create)(void **, void *);
int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    void *fw = dlopen("/System/Library/PrivateFrameworks/Espresso.framework/Espresso", RTLD_NOW);
    f_lib_create lib_create = (f_lib_create)dlsym(fw, "e5rt_program_library_create");
    f_lib_num lib_num = (f_lib_num)dlsym(fw, "e5rt_program_library_get_num_functions");
    f_lib_names lib_names = (f_lib_names)dlsym(fw, "e5rt_program_library_get_function_names");
    f_lib_retain lib_retain = (f_lib_retain)dlsym(fw, "e5rt_program_library_retain_program_function");
    f_opopt_create opopt_create = (f_opopt_create)dlsym(fw, "e5rt_precompiled_compute_op_create_options_create_with_program_function");
    f_opopt_name opopt_name = (f_opopt_name)dlsym(fw, "e5rt_precompiled_compute_op_create_options_set_operation_name");
    f_opopt_alloc opopt_alloc = (f_opopt_alloc)dlsym(fw, "e5rt_precompiled_compute_op_create_options_set_allocate_intermediate_buffers");
    f_op_create op_create = (f_op_create)dlsym(fw, "e5rt_execution_stream_operation_create_precompiled_compute_operation_with_options");

    void *lib = NULL;
    printf("calling library_create...\n");
    int rc = lib_create(&lib, argv[1]);
    printf("library_create = %d lib=%p\n", rc, lib);
    if (rc) return 1;
    uint64_t n = 0; lib_num(lib, &n);
    printf("num_functions = %llu\n", (unsigned long long)n);
    const char **names = calloc(n ? n : 1, sizeof(char *));
    lib_names(lib, n, names);
    for (uint64_t i = 0; i < n; i++) printf("  fn[%llu] %s\n", (unsigned long long)i, names[i]);
    void *fn = NULL;
    rc = lib_retain(lib, names[0], &fn);
    printf("retain = %d fn=%p\n", rc, fn);
    if (rc) return 1;
    void *opopts = NULL;
    printf("calling opopt_create...\n");
    rc = opopt_create(&opopts, fn);
    printf("opopt_create = %d\n", rc);
    if (rc) return 1;
    opopt_name(opopts, names[0]);
    opopt_alloc(opopts, 1);
    void *op = NULL;
    printf("calling op_create...\n");
    rc = op_create(&op, opopts);
    printf("op_create = %d op=%p\n", rc, op);
    return 0;
}
