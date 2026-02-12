# ds_unit_tests

Standalone unit tests for `mla_bwd` DS/TMEM and TS MMA paths.

## Modules

- `ds_tmem`: WG0 softmax/ds write to TMEM and read back
- `dkv_mma`: `TiledMMA_dKV`/`TiledMMA_dKV_RoPE` TS path
- `dq_mma`: `TiledMMA_dQ`/`TiledMMA_dQ_RoPE` TS path with fixed 3-call structure

## Run all

```bash
cd /Users/chenql/Desktop/workspace/operator/test_operator/ds_unit_tests
python run_all_tests.py
```

## Run one module

```bash
cd /Users/chenql/Desktop/workspace/operator/test_operator/ds_unit_tests/ds_tmem
MAX_JOBS=192 python setup.py build_ext --inplace
python test_ds_tmem.py
```
