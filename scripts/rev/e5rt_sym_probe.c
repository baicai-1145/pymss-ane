#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
typedef int e5rt_error_code_t;
int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    void *fw = dlopen("/System/Library/PrivateFrameworks/Espresso.framework/Espresso", RTLD_NOW);
    if (!fw) { printf("dlopen: %s\n", dlerror()); return 1; }
    printf("dlopen ok\n");
    void *s1 = dlsym(fw, "e5rt_program_library_create");
    printf("lib_create sym: %p\n", s1);
    void *s2 = dlsym(fw, "e5rt_precompiled_compute_op_create_options_create_with_program_function");
    printf("opopt sym: %p\n", s2);
    void *s3 = dlsym(fw, "e5rt_execution_stream_operation_create_precompiled_compute_operation_with_options");
    printf("opcreate sym: %p\n", s3);
    void *s4 = dlsym(fw, "e5rt_execution_stream_create");
    printf("stream sym: %p\n", s4);
    return 0;
}
