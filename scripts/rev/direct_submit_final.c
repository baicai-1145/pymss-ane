// direct_submit_final.c — production e5-bundle direct execution:
// library → function → precompiled op → port probe → buffer bind → submit.
// Port names are probed from the E5 symbol table / mlmodel config conventions.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

typedef int e5rt_error_code_t;
#define L(v, sym) do { v = (typeof(v))dlsym(fw, sym); if (!v) v = (typeof(v))dlsym(fw, "_" sym); \
    if (!v) { printf("missing %s\n", sym); return 1; } } while (0)

typedef e5rt_error_code_t (*f_lib_create)(void **, const char *);
typedef e5rt_error_code_t (*f_lib_num)(void *, uint64_t *);
typedef e5rt_error_code_t (*f_lib_names)(void *, uint64_t, const char **);
typedef e5rt_error_code_t (*f_lib_retain)(void *, const char *, void **);
typedef e5rt_error_code_t (*f_opopt_create)(void **, void *);
typedef e5rt_error_code_t (*f_opopt_name)(void *, const char *);
typedef e5rt_error_code_t (*f_opopt_alloc)(void *, int);
typedef e5rt_error_code_t (*f_op_create)(void **, void *);
typedef e5rt_error_code_t (*f_retain_in)(void *, const char *, void **);
typedef e5rt_error_code_t (*f_retain_out)(void *, const char *, void **);
typedef e5rt_error_code_t (*f_port_rel)(void **);
typedef e5rt_error_code_t (*f_buf_alloc)(void **, size_t, int);
typedef e5rt_error_code_t (*f_buf_rel)(void **);
typedef e5rt_error_code_t (*f_buf_ptr)(void *, void **);
typedef e5rt_error_code_t (*f_bind)(void *, void *);
typedef e5rt_error_code_t (*f_stream_create)(void **);
typedef e5rt_error_code_t (*f_stream_enc)(void *, void *);
typedef e5rt_error_code_t (*f_stream_submit)(void *);
typedef e5rt_error_code_t (*f_stream_sync)(void *);

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 3) { printf("usage: %s <bundle> <function>\n", argv[0]); return 1; }
    void *fw = dlopen("/System/Library/PrivateFrameworks/Espresso.framework/Espresso", RTLD_NOW);
    if (!fw) { printf("dlopen: %s\n", dlerror()); return 1; }
    f_lib_create lib_create = 0; f_lib_num lib_num = 0; f_lib_names lib_names = 0;
    f_lib_retain lib_retain = 0; f_opopt_create opopt_create = 0;
    f_opopt_name opopt_name = 0; f_opopt_alloc opopt_alloc = 0; f_op_create op_create = 0;
    f_retain_in retain_in = 0; f_retain_out retain_out = 0; f_port_rel port_rel = 0;
    f_buf_alloc buf_alloc = 0; f_buf_rel buf_rel = 0; f_buf_ptr buf_ptr = 0; f_bind bind = 0;
    f_stream_create stream_create = 0; f_stream_enc stream_enc = 0;
    f_stream_submit stream_submit = 0; f_stream_sync stream_sync = 0;
    L(lib_create, "e5rt_program_library_create");
    L(lib_num, "e5rt_program_library_get_num_functions");
    L(lib_names, "e5rt_program_library_get_function_names");
    L(lib_retain, "e5rt_program_library_retain_program_function");
    L(opopt_create, "e5rt_precompiled_compute_op_create_options_create_with_program_function");
    L(opopt_name, "e5rt_precompiled_compute_op_create_options_set_operation_name");
    L(opopt_alloc, "e5rt_precompiled_compute_op_create_options_set_allocate_intermediate_buffers");
    L(op_create, "e5rt_execution_stream_operation_create_precompiled_compute_operation_with_options");
    L(retain_in, "e5rt_execution_stream_operation_retain_input_port");
    L(retain_out, "e5rt_execution_stream_operation_retain_output_port");
    L(port_rel, "e5rt_io_port_release");
    L(buf_alloc, "e5rt_buffer_object_alloc");
    L(buf_rel, "e5rt_buffer_object_release");
    L(buf_ptr, "e5rt_buffer_object_get_data_ptr");
    L(bind, "e5rt_io_port_bind_buffer_object");
    // stream APIs (name variants)
    stream_create = (f_stream_create)dlsym(fw, "e5rt_execution_stream_create");
    stream_enc = (f_stream_enc)dlsym(fw, "e5rt_execution_stream_encode_operation");
    if (!stream_enc) stream_enc = (f_stream_enc)dlsym(fw, "_e5rt_execution_stream_encode_operation");
    stream_submit = (f_stream_submit)dlsym(fw, "e5rt_execution_stream_submit");
    if (!stream_submit) stream_submit = (f_stream_submit)dlsym(fw, "e5rt_execution_stream_submit_sync");
    if (!stream_submit) stream_submit = (f_stream_submit)dlsym(fw, "_e5rt_execution_stream_submit_sync");
    printf("stream syms: create=%p enc=%p submit=%p\n", stream_create, stream_enc, stream_submit);

    void *lib = NULL;
    int rc = lib_create(&lib, argv[1]);
    if (rc) { printf("library_create=%d\n", rc); return 1; }
    void *fn = NULL;
    rc = lib_retain(lib, argv[2], &fn);
    if (rc) { printf("retain=%d\n", rc); return 1; }
    void *opopts = NULL;
    opopt_create(&opopts, fn);
    opopt_name(opopts, argv[2]);
    opopt_alloc(opopts, 1);
    void *op = NULL;
    rc = op_create(&op, opopts);
    printf("op_create=%d op=%p\n", rc, op);
    if (rc) return 2;

    // probe input/output port names
    const char *cands[] = {"main", "x", "input", "InputFeats", "input_cache_0",
                           "t0", "arg_frame", "in", "data", NULL};
    for (int i = 0; cands[i]; i++) {
        void *port = NULL;
        rc = retain_in(op, cands[i], &port);
        printf("in(%s) rc=%d port=%p\n", cands[i], rc, port);
        if (port) port_rel(&port);
        port = NULL;
        rc = retain_out(op, cands[i], &port);
        if (!rc && port) { printf("OUT PORT FOUND: %s\n", cands[i]); port_rel(&port); }
    }
    // also enumerate what op_create knows: try common output names
    const char *outs[] = {"output", "out", "y", "result", "var", NULL};
    for (int i = 0; outs[i]; i++) {
        void *port = NULL;
        rc = retain_out(op, outs[i], &port);
        if (!rc && port) { printf("OUT: %s ok\n", outs[i]); port_rel(&port); }
    }
    printf("probe done\n");

    // full run: bind buffers to found ports, prepare, encode, submit
    void *in_port = NULL, *out_port = NULL;
    retain_in(op, "input", &in_port);
    retain_out(op, "output", &out_port);
    if (!in_port || !out_port) { printf("port missing\n"); return 3; }
    void *in_buf = NULL, *out_buf = NULL, *in_ptr = NULL, *out_ptr = NULL;
    buf_alloc(&in_buf, 1 << 20, 0);          // 1MB input
    buf_alloc(&out_buf, 1 << 20, 0);         // 1MB output
    buf_ptr(in_buf, &in_ptr); buf_ptr(out_buf, &out_ptr);
    memset(in_ptr, 0, 1 << 16);
    printf("bind in=%d out=%d\n", bind(in_port, in_buf), bind(out_port, out_buf));
    void *prep = dlsym(fw, "e5rt_execution_stream_operation_prepare_op_for_encode");
    if (!prep) prep = dlsym(fw, "_e5rt_execution_stream_operation_prepare_op_for_encode");
    typedef e5rt_error_code_t (*f_prep)(void *);
    printf("prepare=%d\n", ((f_prep)prep)(op));
    void *stream = NULL;
    printf("stream_create=%d\n", stream_create(&stream));
    typedef e5rt_error_code_t (*f_enc)(void *, void *);
    printf("encode=%d\n", ((f_enc)stream_enc)(stream, op));
    // submit_async with a completion block
    typedef e5rt_error_code_t (*f_sub)(void *, void *);
    void *sub = dlsym(fw, "e5rt_execution_stream_submit_async");
    if (!sub) sub = dlsym(fw, "_e5rt_execution_stream_submit_async");
    if (sub) {
        printf("submit_async=%d\n", ((f_sub)sub)(stream, NULL));
    } else {
        printf("no submit_async\n");
    }
    // sample output
    float *o = (float *)out_ptr;
    printf("out[0..3] = %f %f %f %f\n", o[0], o[1], o[2], o[3]);
    printf("FULL RUN ATTEMPTED\n");
    return 0;
}
