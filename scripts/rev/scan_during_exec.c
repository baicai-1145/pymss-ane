// scan_during_exec.c — dlopen'd into python via ctypes BEFORE aneforge runs;
// exposes sar_scan() to scan our process for 0xbeefface at any moment.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>

#define MAGIC 0xBEEFFACE

size_t sar_scan(void) {
    task_t self = mach_task_self();
    mach_vm_address_t addr = 0;
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t binfo;
    mach_msg_type_number_t bcnt = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t obj = MACH_PORT_NULL;
    size_t hits = 0, total = 0, regions = 0;
    while (1) {
        kern_return_t kr = mach_vm_region(self, &addr, &size, VM_REGION_BASIC_INFO_64,
                                          (vm_region_info_t)&binfo, &bcnt, &obj);
        if (kr != KERN_SUCCESS) break;
        if ((binfo.protection & 2) && size < (1ULL << 32)) {
            regions++; total += size;
            const mach_vm_size_t CH = 1 << 22;
            for (mach_vm_size_t off = 0; off < size; off += CH) {
                mach_vm_size_t n = (size - off < CH) ? (size - off) : CH;
                vm_offset_t data = 0;
                mach_msg_type_number_t cnt = 0;
                if (vm_read(self, addr + off, (mach_msg_type_number_t)n, &data, &cnt) != KERN_SUCCESS)
                    continue;
                for (mach_msg_type_number_t i = 0; i + 4 <= cnt; i += 4) {
                    uint32_t m; memcpy(&m, (void *)(data + i), 4);
                    if (m == MAGIC) {
                        printf("[hit] %#llx\n", (unsigned long long)(addr + off + i));
                        char path[128];
                        snprintf(path, sizeof(path), "/tmp/sde_hit_%llx_%zu.bin",
                                 (unsigned long long)(addr + off + i), hits);
                        FILE *f = fopen(path, "wb");
                        if (f) {
                            size_t cap = cnt - i; if (cap > (2 << 20)) cap = 2 << 20;
                            fwrite((void *)(data + i), 1, cap, f); fclose(f);
                        }
                        hits++;
                    }
                }
                vm_deallocate(mach_task_self(), data, cnt);
            }
        }
        addr += size ? size : 0x1000;
    }
    fprintf(stderr, "[sde] regions=%zu total=%.1fMB hits=%zu\n", regions, total / 1048576.0, hits);
    return hits;
}
