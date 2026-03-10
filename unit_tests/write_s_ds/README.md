# write_s_ds

This unit test isolates the `64daee0334475278d49674e91fa796392f6c13a4` regression into two checks:

1. Scalar `SmemLayoutS` writes versus the 128-bit vectorized shared-memory write formula.
2. Correct half-tile masking versus the buggy loop that only masks the first 16 floats after casting `float2[16]` to `float*`.

Expected result:

- Scalar and vectorized writes match exactly.
- The buggy path differs only in columns `16:32` and `48:64`.

That means the regression is not the `sS_base + vec * 64 * 8` address formula itself. The bad data comes from only masking half of each 32-float half-tile before writing `S/dS` to shared memory.

Build and run:

```bash
cd /Users/chenql/Desktop/workspace/operator/test_operator/unit_tests/write_s_ds
MAX_JOBS=192 python setup.py build_ext --inplace
python test_write_s_ds.py
```
