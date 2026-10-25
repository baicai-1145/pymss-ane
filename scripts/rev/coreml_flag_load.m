// t18_load.m — load a CoreML model with MLModelConfiguration.e5rtCustomANECompilerOptions
// to steer the runtime e5rt compile (ANE flags channel).
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/message.h>
#include <dlfcn.h>

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 2) { printf("usage: %s <mlmodel/mlpackage> [anec_flags]\n", argv[0]); return 1; }
        NSString *path = [NSString stringWithUTF8String:argv[1]];
        MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        if (argc > 2 && strlen(argv[2])) {
            NSString *flags = [NSString stringWithUTF8String:argv[2]];
            @try {
                ((void (*)(id, SEL, id))objc_msgSend)(cfg, @selector(setE5rtCustomANECompilerOptions:), flags);
                NSLog(@"[t18] set e5rtCustomANECompilerOptions = %@", flags);
            } @catch (NSException *e) {
                NSLog(@"[t18] KVC fallback: %@", e);
                [cfg setValue:flags forKey:@"e5rtCustomANECompilerOptions"];
            }
        }
        NSError *err = nil;
        NSURL *url = [NSURL fileURLWithPath:path];
        // runtime compile (the e5rt line) if given a source mlmodel/mlpackage
        if ([path hasSuffix:@"mlmodel"] || [path hasSuffix:@"mlpackage"]) {
            NSURL *compiled = [MLModel compileModelAtURL:url error:&err];
            if (!compiled) { NSLog(@"[t18] compileModel FAILED: %@", err); return 3; }
            NSLog(@"[t18] runtime-compiled to: %@", compiled);
            url = compiled;
        }
        MLModel *model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
        if (!model) { NSLog(@"[t18] load FAILED: %@", err); return 2; }
        NSLog(@"[t18] model loaded OK: %@", model);
        // run one prediction to force compile + ANE exec
        MLDictionaryFeatureProvider *inp = nil;
        err = nil;
        MLModelDescription *desc = model.modelDescription;
        NSString *inName = desc.inputDescriptionsByName.allKeys.firstObject;
        NSString *outName = desc.outputDescriptionsByName.allKeys.firstObject;
        NSArray *shape = desc.inputDescriptionsByName[inName].multiArrayConstraint.shape;
        NSMutableArray *dims = [NSMutableArray array];
        for (NSNumber *d in shape) [dims addObject:d];
        NSLog(@"[t18] input %@ shape %@ output %@", inName, dims, outName);
        NSError *perr = nil;
        MLMultiArray *ma = [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeFloat16 error:&perr];
        if (!ma) { NSLog(@"[t18] ma err %@", perr); return 4; }
        MLDictionaryFeatureProvider *input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{inName: ma} error:&perr];
        if (!input) { NSLog(@"[t18] provider err %@", perr); return 5; }
        id out = [model predictionFromFeatures:input error:&perr];
        NSLog(@"[t18] prediction %@ err=%@", out ? @"OK" : @"FAIL", perr);
        return 0;
    }
}
