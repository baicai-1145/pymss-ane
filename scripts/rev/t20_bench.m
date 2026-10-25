// t20_bench.m — variant bench with p50 + flag-landing verification support.
// Loads mlmodelc dir (env T20_FLAGS injected via net properties beforehand),
// runs warmup + N timed predictions, prints p50/p90/mean.
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/message.h>

int main(int argc, char **argv) {
    @autoreleasepool {
        NSString *path = [NSString stringWithUTF8String:argv[1]];
        int N = argc > 2 ? atoi(argv[2]) : 60;
        int WARM = argc > 3 ? atoi(argv[3]) : 15;
        MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
        const char *cu = getenv("T20_CU");
        cfg.computeUnits = (cu && strcmp(cu, "cpu") == 0) ? MLComputeUnitsCPUOnly :
                           (cu && strcmp(cu, "ane") == 0) ? MLComputeUnitsCPUAndNeuralEngine :
                           MLComputeUnitsCPUAndNeuralEngine;
        NSError *err = nil;
        MLModel *model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:path] configuration:cfg error:&err];
        if (!model) { NSLog(@"[t20] load fail %@", err); return 2; }
        MLModelDescription *desc = model.modelDescription;
        NSString *inName = desc.inputDescriptionsByName.allKeys.firstObject;
        NSString *outName = desc.outputDescriptionsByName.allKeys.firstObject;
        NSArray *shape = desc.inputDescriptionsByName[inName].multiArrayConstraint.shape;
        MLMultiArray *ma = [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeFloat32 error:&err];
        if (!ma) { NSLog(@"[t20] input alloc fail %@", err); return 3; }
        float *p = ma.dataPointer;
        for (NSUInteger i = 0; i < ma.count; i++) p[i] = (float)((i * 2654435761u % 2000) - 1000) / 1000.0f;
        MLDictionaryFeatureProvider *input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{inName: ma} error:&err];
        id out;
        for (int i = 0; i < WARM; i++) out = [model predictionFromFeatures:input error:&err];
        if (!out) { NSLog(@"[t20] warmup predict fail %@", err); return 4; }
        NSMutableArray<NSNumber *> *times = [NSMutableArray array];
        for (int i = 0; i < N; i++) {
            CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
            out = [model predictionFromFeatures:input error:&err];
            [times addObject:@((CFAbsoluteTimeGetCurrent() - t0) * 1000.0)];
        }
        [times sortUsingSelector:@selector(compare:)];
        double p50 = times[N/2].doubleValue;
        double p90 = times[(int)(N*0.9)].doubleValue;
        double mean = 0; for (NSNumber *t in times) mean += t.doubleValue; mean /= N;
        // dump output hash for numeric comparison across variants
        MLMultiArray *oma = [out featureValueForName:outName].multiArrayValue;
        // simple checksum: sum of first 1024 floats
        float chk = 0; const float *op = oma.dataPointer;
        for (int i = 0; i < 1024 && i < (int)oma.count; i++) chk += op[i * 977];
        NSLog(@"[t20] p50=%.4f p90=%.4f mean=%.4f N=%d chk=%.6f", p50, p90, mean, N, chk);
        return 0;
    }
}
