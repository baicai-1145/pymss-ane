// t22_blockhook.c — runtime block-invoke replacement for transform_softmax bypass.
//
// Finding (task-22): Espresso's transform_softmax blocks live on the heap
// (0x16d... RW), pointers found at runtime by scanning for the static
// invoke addresses (0x1a0906d44 invoke_2 / 0x1a0906644 invoke_1 + slide).
// Replacing invoke_2 with a no-op-ish trampoline disables the softmax
// elementwise decomposition at the pass level.
//
// Usage: ctypes.CDLL(...); t22_hook() after Espresso is loaded, BEFORE
// aneforge compile. Restore with t22_unhook().
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach-o/dyld.h>

static uintptr_t g_inv2_runtime = 0;
static void **g_slots[16];
static int g_nslots = 0;

// Replacement invoke: mimic block signature (block ctx, layer) but skip
// the decomposition — return as if nothing matched.
static long t22_replacement_invoke(void *blk, void *arg) {
    (void)blk; (void)arg;
    return 0;  // "no transform"
}

int t22_hook(void) {
    intptr_t slide = 0;
    for (uint32_t i = 0; i < _dyld_image_count(); i++) {
        const char *n = _dyld_get_image_name(i);
        if (n && strstr(n, "Espresso.framework")) {
            slide = _dyld_get_image_vmaddr_slide(i);
            break;
        }
    }
    if (!slide) { fprintf(stderr, "[hook] espresso not loaded\n"); return -1; }
    g_inv2_runtime = 0x1a0906d44 + slide;
    task_t self = mach_task_self();
    mach_vm_address_t addr = 0; mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t bi;
    mach_msg_type_number_t bc = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t obj = MACH_PORT_NULL;
    g_nslots = 0;
    while (g_nslots < 16) {
        kern_return_t kr = mach_vm_region(self, &addr, &size, VM_REGION_BASIC_INFO_64,
                                          (vm_region_info_t)&bi, &bc, &obj);
        if (kr != KERN_SUCCESS) break;
        // heap regions only (0x16... stack is per-thread; transform blocks are heap/malloc
        // but may share arena with stack pages — take all RW)
        if ((bi.protection & 2) && size < (1ULL<<30)) {
            vm_offset_t data = 0; mach_msg_type_number_t cnt = 0;
            mach_vm_size_t rd = size > (1<<24) ? (1<<24) : size;
            if (vm_read(self, addr, (mach_msg_type_number_t)rd, &data, &cnt) == KERN_SUCCESS) {
                for (mach_msg_type_number_t o = 0; o + 8 <= cnt; o += 8) {
                    uintptr_t v; memcpy(&v, (void *)(data + o), 8);
                    if (v == g_inv2_runtime) {
                        void **slot = (void **)(addr + o);
                        g_slots[g_nslots++] = slot;
                        *slot = (void *)t22_replacement_invoke;
                    }
                }
                vm_deallocate(mach_task_self(), data, cnt);
            }
        }
        addr += size ? size : 0x1000;
    }
    fprintf(stderr, "[hook] replaced %d invoke_2 slots\n", g_nslots);
    return g_nslots;
}

int t22_unhook(void) {
    for (int i = 0; i < g_nslots; i++) *g_slots[i] = (void *)g_inv2_runtime;
    fprintf(stderr, "[hook] restored %d slots\n", g_nslots);
    return 0;
}
