# ESPRESSO_PRIVATE_* env probes — findings (task-17)

## Method
Fresh-compile matrix (CH=96/104/112/120 conv) x {baseline, IGNORE_ANE_PRECOMPILED,
ALWAYS_COMPRESSED, TUNELOG} + same-input re-runs + poison tests.

## Findings
1. **E5 output identical across all envs** (md5 match per shape) — none of the
   three envs changes the compiled artifact on the e5rt path.
2. **anehash h1 constant per graph (env-independent); h2 = per-compile-session
   nonce** — two consecutive no-env runs of the same input produce different h2.
   h2 is NOT a content hash; it cannot be used for integrity verification.
3. **anehash is never verified on the e5rt path**: poisoning it to AAAA_BBBB
   in the live cache still compiles + executes OK (verifyBundleAtPath belongs
   to the disk _ANEModel line only).
4. **E5 is rewritten by every compile** — a byte-patched E5 does not survive
   the next compile() call (regenerated in place). Disk E5 patching has a
   zero survival window on this path.

## Implications
- Channel a (disk bundle patch) is doubly dead: hash gate doesn't exist, but
  the artifact itself is regenerated each compile.
- The only compiler-input surface we already control = MIL (aneforge's native
  input). IR surgery (dump-IR line) and MIL surgery are the same layer.
- aned-side (sudo spare) remains the only container-byte surgery surface.
