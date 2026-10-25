// task-16 final: load compiled mlmodelc with MLModelConfiguration.e5rtCustomANECompilerOptions
// and force ANE compile via prediction. Usage: cfg_compile <flagstring> <model.mlmodelc> <x-proto-input>
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/message.h>
#include <stdio.h>

@interface MLModelConfiguration (Private)
- (void)setE5rtCustomANECompilerOptions:(NSString *)s;
- (NSString *)e5rtCustomANECompilerOptions;
@end

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 3) { fprintf(stderr, "usage: %s <flags> <model.mlmodelc>\n", argv[0]); return 1; }
    NSString *flag = [NSString stringWithUTF8String:argv[1]];
    NSString *modelPath = [NSString stringWithUTF8String:argv[2]];

    MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
    if (![flag isEqualToString:@"-"]) ((void (*)(id, SEL, id))objc_msgSend)(cfg, @selector(setE5rtCustomANECompilerOptions:), flag);
    cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    NSLog(@"[cfg] options='%@'", ((id (*)(id, SEL))objc_msgSend)(cfg, @selector(e5rtCustomANECompilerOptions)));

    NSURL *url = [NSURL fileURLWithPath:modelPath];
    NSError *err = nil;
    id m = ((id (*)(Class, SEL, id, id, NSError **))objc_msgSend)((id)[MLModel class], @selector(modelWithContentsOfURL:configuration:error:), url, cfg, &err);
    if (!m) { NSLog(@"[cfg] load failed: %@", err); return 1; }
    NSLog(@"[cfg] loaded OK");

    // build input: 480x256x256 MLMultiArray fp16
    NSArray *shape = @[@480, @256, @256];
    MLMultiArray *x = [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeFloat16 error:&err];
    if (!x) { NSLog(@"[cfg] input alloc failed: %@", err); return 1; }
    MLDictionaryFeatureProvider *prov = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x": x} error:&err];
    if (!prov) { NSLog(@"[cfg] provider failed: %@", err); return 1; }
    id out = [m predictionFromFeatures:prov options:[[MLPredictionOptions alloc] init] error:&err];
    if (!out) { NSLog(@"[cfg] predict failed: %@", err); return 1; }
    NSLog(@"[cfg] predict OK");
    return 0;
  }
}
