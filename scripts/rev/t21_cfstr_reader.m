#import <Foundation/Foundation.h>
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <string.h>
int main(void) {
    // Ensure Espresso is mapped
    dlopen("/System/Library/PrivateFrameworks/Espresso.framework/Espresso", RTLD_NOW);
    for (uint32_t i = 0; i < _dyld_image_count(); i++) {
        const char *n = _dyld_get_image_name(i);
        if (!n || !strstr(n, "Espresso")) continue;
        intptr_t slide = _dyld_get_image_vmaddr_slide(i);
        const struct mach_header *h = _dyld_get_image_header(i);
        if (((uint64_t)h & 0xfff) && ((uint64_t)h > 0x70000000000ULL)) continue; // skip caches? no — read anyway
        printf("image: %s slide=%#lx hdr=%p\n", n, (long)slide, (void *)h);
        // static addr = runtime - slide. We want runtime addr for static 0x1ea890428:
        uint64_t runtime = 0x1ea890428 + slide;
        // CFString at that addr:
        void *cf = (void *)runtime;
        // read CFString struct: {isa, flags, ptr, length}
        uint64_t *q = (uint64_t *)cf;
        printf("cf@%#llx struct: isa=%#llx flags=%#llx ptr=%#llx len=%llu -> \'%s\'\n",
               (unsigned long long)runtime, (unsigned long long)q[0], (unsigned long long)q[1],
               (unsigned long long)q[2], (unsigned long long)q[3], (char *)(uintptr_t)q[2]);
        uint64_t addrs[2] = {0x1ea890448ULL, 0x1ea890408ULL};
        for (int k = 0; k < 2; k++) { uint64_t a = addrs[k];
            uint64_t *q = (uint64_t *)(a + slide);
            printf("cf@%#llx -> \'%s\' (len %llu)\n", (unsigned long long)a,
                   (char *)(uintptr_t)q[2], (unsigned long long)q[3]);
        }
        break;
    }
    return 0;
}
