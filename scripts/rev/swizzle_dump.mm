// ObjC swizzle: after compileWithQoS completes, walk the model's ivars + properties
// hunting for NSData >= 4KB with Mach-O magics (0xfeedface/beefface) -> dump.
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static void dump_nsdata(id obj, const char *tag) {
  if (![obj isKindOfClass:[NSData class]]) return;
  NSData *d = obj;
  if (d.length < 4096) return;
  const unsigned char *p = (const unsigned char *)d.bytes;
  unsigned int m = *(const unsigned int *)p;
  if (m == 0xfeedface || m == 0xfeedfacf || m == 0xbeefface || m == 0xcafebabe) {
    const char *dir = getenv("ANEC_DUMP_DIR");
    char path[512];
    snprintf(path, sizeof path, "%s/swz_%s_MachO_%x_%zu.bin", dir ? dir : "/tmp", tag, m, d.length);
    [d writeToFile:[NSString stringWithUTF8String:path] atomically:YES];
    printf("[swz] MachO %x (%zu bytes) via %s -> %s\n", m, d.length, tag, path);
  }
}

static void scan_object(id root, const char *tag);

static void describe_object(id root, const char *tag) {
  if (!root) return;
  Class c = object_getClass(root);
  int depth = 0;
  while (c && depth < 5) {
    unsigned int n = 0;
    Ivar *ivs = class_copyIvarList(c, &n);
    for (unsigned int i = 0; i < n; i++) {
      const char *ty = ivar_getTypeEncoding(ivs[i]);
      const char *nm = ivar_getName(ivs[i]);
      if (ty && ty[0] == '@') {
        id v = object_getIvar(root, ivs[i]);
        if (v) printf("[swz-describe] %s.%s : %@ (%zu bytes)\n", tag, nm ? nm : "?",
                      [NSString stringWithFormat:@"%@", [v class]], (size_t)[v respondsToSelector:@selector(length)] ? [(NSData *)v length] : 0);
      } else if (ty) {
        printf("[swz-describe] %s.%s : %s\n", tag, nm ? nm : "?", ty);
      }
    }
    if (ivs) free(ivs);
    c = class_getSuperclass(c);
    depth++;
  }
}

static void scan_object(id root, const char *tag) {
  if (!root) return;
  Class c = object_getClass(root);
  int depth = 0;
  while (c && depth < 4) {
    unsigned int n = 0;
    Ivar *ivs = class_copyIvarList(c, &n);
    for (unsigned int i = 0; i < n; i++) {
      const char *ty = ivar_getTypeEncoding(ivs[i]);
      const char *nm = ivar_getName(ivs[i]);
      if (!ty) continue;
      if (ty[0] == '@') {
        id v = object_getIvar(root, ivs[i]);
        if (v && v != root) {
          char sub[128];
          snprintf(sub, sizeof sub, "%s.%s", tag, nm ? nm : "?");
          dump_nsdata(v, sub);
          if ([v isKindOfClass:[NSArray class]] && [(NSArray *)v count] < 64) {
            for (id e in (NSArray *)v) dump_nsdata(e, sub);
          } else if ([v isKindOfClass:[NSDictionary class]] && [(NSDictionary *)v count] < 64) {
            for (id e in [(NSDictionary *)v allValues]) dump_nsdata(e, sub);
          }
        }
      }
    }
    if (ivs) free(ivs);
    c = class_getSuperclass(c);
    depth++;
  }
}

static IMP orig_compile = NULL;
static BOOL swz_compile(id self, SEL _cmd, unsigned int qos, id opts, NSError **err) {
  BOOL rc = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))orig_compile)(self, _cmd, qos, opts, err);
  printf("[swz] compileWithQoS rc=%d on %s\n", rc, class_getName(object_getClass(self)));
  if (rc) {
    Class cc = object_getClass(self);
    Ivar ivu = class_getInstanceVariable(cc, "_modelURL");
    if (ivu) { id u = object_getIvar(self, ivu); printf("[swz] _modelURL = %@\n", u); }
    Ivar ivm = class_getInstanceVariable(cc, "_model");
    if (ivm) { id mm = object_getIvar(self, ivm);
      Class cm = object_getClass(mm);
      Ivar ivu2 = class_getInstanceVariable(cm, "_modelURL");
      if (ivu2) printf("[swz] _model._modelURL = %@\n", object_getIvar(mm, ivu2));
      Ivar ivk = class_getInstanceVariable(cm, "_key");
      if (ivk) printf("[swz] _model._key = %@\n", object_getIvar(mm, ivk));
      Ivar ivatt = class_getInstanceVariable(cm, "_modelAttributes");
      if (ivatt) { id att = object_getIvar(mm, ivatt); printf("[swz] _model._modelAttributes = %@\n", att); }
    }
    describe_object(self, "model");
    scan_object(self, "model");
    Class c0 = object_getClass(self);
    Ivar ivModel = class_getInstanceVariable(c0, "_model");
    Ivar ivAttr = class_getInstanceVariable(c0, "_modelAttributes");
    if (ivModel) { id m = object_getIvar(self, ivModel); printf("[swz-deep] _model = %s\n", m ? class_getName(object_getClass(m)) : "nil"); if (m) describe_object(m, "model._model"); if (m) scan_object(m, "model._model"); }
    if (ivAttr) { id a = object_getIvar(self, ivAttr); printf("[swz-deep] _modelAttributes = %s\n", a ? class_getName(object_getClass(a)) : "nil"); if (a) describe_object(a, "model._modelAttributes"); if (a) scan_object(a, "model._modelAttributes"); }
  }
  return rc;
}

__attribute__((constructor))
static void setup(void) {
  dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_LAZY);
  Class cls = NSClassFromString(@"_ANEInMemoryModel");
  if (!cls) { printf("[swz] _ANEInMemoryModel not found (load order?)\n"); return; }
  Method m = class_getInstanceMethod(cls, @selector(compileWithQoS:options:error:));
  if (!m) { printf("[swz] method not found\n"); return; }
  orig_compile = method_getImplementation(m);
  method_setImplementation(m, (IMP)swz_compile);
  printf("[swz] swizzle installed on _ANEInMemoryModel\n");
}


// ---- decode-path swizzles ----
static IMP orig_procdata_coder = NULL;
static id swz_procdata_coder(id self, SEL _cmd, NSCoder *coder) {
  id rc = ((id (*)(id, SEL, id))orig_procdata_coder)(self, _cmd, coder);
  printf("[swz] _ANEProcedureData initWithCoder done\n");
  describe_object(rc, "procdata");
  scan_object(rc, "procdata");
  return rc;
}

static void swizzle_named(const char *clsname, SEL sel, IMP *orig, IMP newimp) {
  Class cls = NSClassFromString([NSString stringWithUTF8String:clsname]);
  if (!cls) { printf("[swz] class %s not found\n", clsname); return; }
  Method m = class_getInstanceMethod(cls, sel);
  if (!m) { printf("[swz] %s method missing\n", clsname); return; }
  *orig = method_getImplementation(m);
  method_setImplementation(m, newimp);
  printf("[swz] %s swizzled\n", clsname);
}

static IMP orig_model_coder = NULL;
static id swz_model_coder(id self, SEL _cmd, NSCoder *coder) {
  id rc = ((id (*)(id, SEL, id))orig_model_coder)(self, _cmd, coder);
  printf("[swz] _ANEModel initWithCoder done\n");
  describe_object(rc, "aned_model");
  scan_object(rc, "aned_model");
  return rc;
}
