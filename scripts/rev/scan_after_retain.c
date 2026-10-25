// scan_after_retain.c — load a bundle via e5rt, retain function, then scan
// OUR OWN process memory for 0xbeefface before Espresso frees anything.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>

typedef int e5rt_error_code_t;
typedef e5rt_error_code_t (*f_lib_create)(void **, const char *);
typedef e5rt_error_code_t (*f_lib_retain)(void *, const char *, void **);

#define MAGIC 0xBEEFFACE

static size_t scan_region(task_t t, mach_vm_address_t addr, mach_vm_size_t size) {
    size_t hits = 0;
    const mach_vm_size_t CH = 1 << 22;
    for (mach_vm_size_t off = 0; off < size; off += CH) {
        mach_vm_size_t n = (size - off < CH) ? (size - off) : CH;
        vm_offset_t data = 0;
        mach_msg_type_number_t cnt = 0;
        if (vm_read(t, addr + off, (mach_msg_type_number_t)n, &data, &cnt) != KERN_SUCCESS)
            continue;
        for (mach_msg_type_number_t i = 0; i + 4 <= cnt; i += 4) {
            uint32_t m;
            memcpy(&m, (void *)(data + i), 4);
            if (m == MAGIC) {
                printf("[hit] %#llx +%u\n", (unsigned long long)(addr + off + i), i);
                char path[128];
                snprintf(path, sizeof(path), "/tmp/sar_hit_%llx.bin",
                         (unsigned long long)(addr + off + i));
                FILE *f = fopen(path, "wb");
                if (f) {
                    size_t cap = cnt - i;
                    if (cap > (1 << 20)) cap = 1 << 20;
                    fwrite((void *)(data + i), 1, cap, f);
                    fclose(f);
                }
                hits++;
                if (hits > 8) { vm_deallocate(mach_task_self(), data, cnt); break; }
            }
        }
        vm_deallocate(mach_task_self(), data, cnt);
        if (hits > 8) break;
    }
    return hits;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) { printf("usage: %s <bundle>\n", argv[0]); return 1; }
    void *fw = dlopen("/System/Library/PrivateFrameworks/Espresso.framework/Espresso", RTLD_NOW);
    f_lib_create lib_create = (f_lib_create)dlsym(fw, "e5rt_program_library_create");
    f_lib_retain lib_retain = (f_lib_retain)dlsym(fw, "e5rt_program_library_retain_program_function");
    void *lib = NULL;
    int rc = lib_create(&lib, argv[1]);
    printf("library_create=%d lib=%p\n", rc, lib);
    if (rc) return 1;
    void *fn = NULL;
    rc = lib_retain(lib, "main", &fn);
    printf("retain=%d fn=%p\n", rc, fn);
    // also create the op (device-bound step pulls everything needed)
    typedef e5rt_error_code_t (*f_opopt_create)(void **, void *);
    typedef e5rt_error_code_t (*f_opopt_name)(void *, const char *);
    typedef e5rt_error_code_t (*f_opopt_alloc)(void *, int);
    typedef e5rt_error_code_t (*f_op_create)(void **, void *);
    f_opopt_create oc = (f_opopt_create)dlsym(fw, "e5rt_precompiled_compute_op_create_options_create_with_program_function");
    f_opopt_name on = (f_opopt_name)dlsym(fw, "e5rt_precompiled_compute_op_create_options_set_operation_name");
    f_opopt_alloc oa = (f_opopt_alloc)dlsym(fw, "e5rt_precompiled_compute_op_create_options_set_allocate_intermediate_buffers");
    f_op_create opr = (f_op_create)dlsym(fw, "e5rt_execution_stream_operation_create_precompiled_compute_operation_with_options");
    void *opopts = NULL;
    if (oc) oc(&opopts, fn);
    if (on && opopts) on(opopts, "main");
    if (oa && opopts) oa(opopts, 1);
    void *op = NULL;
    int orc = opr ? opr(&op, opopts) : -1;
    printf("op_create=%d op=%p\n", orc, op);
    printf("--- scan after retain+op_create ---\n");

    task_t self = mach_task_self();
    mach_vm_address_t addr = 0;
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t binfo;
    mach_msg_type_number_t bcnt = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t obj = MACH_PORT_NULL;
    size_t total = 0, hits = 0, regions = 0;
    while (1) {
        kern_return_t kr = mach_vm_region(self, &addr, &size, VM_REGION_BASIC_INFO_64,
                                          (vm_region_info_t)&binfo, &bcnt, &obj);
        if (kr != KERN_SUCCESS) break;
        if ((binfo.protection & 2) && size < (1ULL << 32)) {
            regions++;
            total += size;
            hits += scan_region(self, addr, size);
        } else if (regions == 0 && addr > 0x100000000ULL) {
            printf("[skip] %#llx+%llu prot=%d\n", (unsigned long long)addr,
                   (unsigned long long)size, binfo.protection);
        }
        addr += size ? size : 0x1000;
        if (hits > 8) break;
    }
    printf("scanned %zu regions, %.1f MB, hits=%zu\n", regions, total / 1048576.0, hits);
    return hits ? 0 : 2;
}
