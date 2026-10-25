// task-14: netplist compile with NON-EMPTY optionsPlist — schema: {KEY: NSArray<NSData>} per
// _ANEHashEncoding (hexStringForDataArray: count + per-element bytes).
// Usage: opts_probe net.plist [weights...] [--opt K=V]...
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 2) { fprintf(stderr, "usage: %s net.plist [weights...] [--opt K=V]\n", argv[0]); return 1; }
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_LAZY);

    NSString *netPath = [NSString stringWithUTF8String:argv[1]];
    NSMutableDictionary *weights = [NSMutableDictionary dictionary];
    NSMutableDictionary *opts = [NSMutableDictionary dictionary];
    for (int i = 2; i < argc; i++) {
      std::string a = argv[i];
      if (a == "--opt" && i + 1 < argc) {
        NSString *kv = [NSString stringWithUTF8String:argv[++i]];
        NSArray *parts = [kv componentsSeparatedByString:@"="];
        NSString *k = parts[0];
        NSString *v = parts.count >= 2 ? parts[1] : @"YES";
        opts[k] = @[ [v dataUsingEncoding:NSUTF8StringEncoding] ];
        continue;
      }
      NSString *p = [NSString stringWithUTF8String:a.c_str()];
      NSData *d = [NSData dataWithContentsOfFile:p];
      if (d) weights[p.lastPathComponent] = @[ d ];
    }
    NSData *netData = [NSData dataWithContentsOfFile:netPath];
    if (!netData) { fprintf(stderr, "cannot read net plist\n"); return 2; }

    NSArray *before = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:@"." error:nil];

    Class desc_cls = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class model_cls = NSClassFromString(@"_ANEInMemoryModel");
    id desc = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
        desc_cls, @selector(modelWithNetworkDescription:weights:optionsPlist:), netData, weights, opts.count ? opts : nil);
    printf("descriptor: %s\n", desc ? "OK" : "FAILED");
    if (!desc) return 3;
    id model = ((id (*)(Class, SEL, id))objc_msgSend)(model_cls, @selector(inMemoryModelWithDescriptor:), desc);
    printf("model: %s\n", model ? "OK" : "FAILED");
    if (!model) return 3;

    NSString *lp = ((id (*)(id, SEL))objc_msgSend)(model, @selector(localModelPath));
    if ([lp isKindOfClass:[NSString class]] && [lp length] > 0) {
      [[NSFileManager defaultManager] createDirectoryAtPath:lp withIntermediateDirectories:YES attributes:nil error:nil];
      [netData writeToFile:[lp stringByAppendingPathComponent:@"net.plist"] atomically:YES];
      for (NSString *wname in weights) {
        NSData *d = weights[wname][0];
        [d writeToFile:[lp stringByAppendingPathComponent:wname] atomically:YES];
      }
    }

    NSError *err = nil;
    BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, @selector(compileWithQoS:options:error:), 33, @{}, &err);
    printf("compile: %s %s\n", ok ? "OK" : "FAILED", err ? [[err description] UTF8String] : "");
    if (!ok) return 4;

    NSArray *after = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:@"." error:nil];
    NSMutableArray *news = [NSMutableArray array];
    for (NSString *f in after) if (![before containsObject:f]) [news addObject:f];
    printf("new files in cwd: %s\n", news.count ? [[news componentsJoinedByString:@", "] UTF8String] : "(none)");
    printf("{\"status\":\"ok\"}\n");
  }
  return 0;
}
