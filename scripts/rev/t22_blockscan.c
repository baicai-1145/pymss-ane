// t22_blockscan.dylib — load into python via ctypes; scans our process for
// pointers to transform_softmax block_invoke functions (Espresso loaded).
// Prints block descriptor locations for runtime-invoke-patch feasibility.
#include <stdio.h>
#include <string.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach-o/dyld.h>
#include <dlfcn.h>

__attribute__((constructor))
static void boot(void) { /* nothing; call scan() explicitly */ }

int t22_scan(void) {
    intptr_t slide = 0;
    for (uint32_t i = 0; i < _dyld_image_count(); i++) {
        const char *n = _dyld_get_image_name(i);
        if (n && strstr(n, "Espresso.framework")) {
            slide = _dyld_get_image_vmaddr_slide(i);
            break;
        }
    }
    fprintf(stderr, "[t22] espresso slide=%#lx\n", (long)slide);
    if (!slide) return 1;
    uintptr_t inv2 = 0x1a0906d44 + slide;
    uintptr_t inv1 = 0x1a0906644 + slide;
    fprintf(stderr, "[t22] invoke_2 runtime=%#lx invoke_1=%#lx\n",
            (unsigned long)inv2, (unsigned long)inv1);
    task_t self = mach_task_self();
    mach_vm_address_t addr = 0; mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t bi;
    mach_msg_type_number_t bc = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t obj = MACH_PORT_NULL;
    int found = 0;
    while (1) {
        kern_return_t kr = mach_vm_region(self, &addr, &size, VM_REGION_BASIC_INFO_64,
                                          (vm_region_info_t)&bi, &bc, &obj);
        if (kr != KERN_SUCCESS) break;
        if ((bi.protection & 2) && size < (1ULL<<30)) {
            vm_offset_t data = 0; mach_msg_type_number_t cnt = 0;
            if (vm_read(self, addr, (mach_msg_type_number_t)(size > (1<<24) ? (1<<24) : size),
                        &data, &cnt) == KERN_SUCCESS) {
                for (mach_msg_type_number_t o = 0; o + 8 <= cnt; o += 8) {
                    uintptr_t v; memcpy(&v, (void *)(data + o), 8);
                    if (v == inv2 || v == inv1) {
                        fprintf(stderr, "[t22] block-ref @%#llx -> invoke%s (prot=%d)\n",
                                (unsigned long long)(addr + o), v == inv2 ? "_2" : "_1", bi.protection);
                        if (++found > 5) { vm_deallocate(mach_task_self(), data, cnt); return found; }
                    }
                }
                vm_deallocate(mach_task_self(), data, cnt);
            }
        }
        addr += size ? size : 0x1000;
    }
    fprintf(stderr, "[t22] scan done found=%d\n", found);
    return found;
}
