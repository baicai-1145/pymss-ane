#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/message.h>

static MLMultiArray *predict(NSString *path, MLMultiArray *seed, NSError **err) {
    MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
    cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    MLModel *model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:path] configuration:cfg error:err];
    if (!model) return nil;
    MLModelDescription *desc = model.modelDescription;
    NSString *inName = desc.inputDescriptionsByName.allKeys.firstObject;
    NSString *outName = desc.outputDescriptionsByName.allKeys.firstObject;
    MLDictionaryFeatureProvider *input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{inName: seed} error:err];
    id out = [model predictionFromFeatures:input error:err];
    if (!out) return nil;
    return [out featureValueForName:outName].multiArrayValue;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        NSArray *shape = @[@512, @64, @64];
        MLMultiArray *seed = [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeFloat32 error:nil];
        float *p = seed.dataPointer;
        for (NSUInteger i = 0; i < seed.count; i++) p[i] = (float)((i * 2654435761u % 1000) - 500) / 500.0f;
        NSError *err = nil;
        MLMultiArray *a = predict([NSString stringWithUTF8String:argv[1]], seed, &err);
        if (!a) { NSLog(@"A fail %@", err); return 1; }
        for (int i = 2; i < argc; i++) {
            err = nil;
            MLMultiArray *b = predict([NSString stringWithUTF8String:argv[i]], seed, &err);
            if (!b) { NSLog(@"%@ FAIL %@", @(argv[i]), err); continue; }
            double maxd = 0; NSUInteger nd = 0;
            const double *pa = a.dataPointer, *pb = b.dataPointer;
            for (NSUInteger j = 0; j < a.count; j++) {
                double d = fabs(pa[j] - pb[j]);
                if (d > 0) nd++;
                if (d > maxd) maxd = d;
            }
            NSLog(@"[num] %@ vs base: maxdiff=%.3e ndiff=%zu/%zu", @(argv[i]), maxd, nd, (NSUInteger)a.count);
        }
    }
    return 0;
}
