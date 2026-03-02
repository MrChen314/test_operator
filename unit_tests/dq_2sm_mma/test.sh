git fetch origin btopk64
git reset --hard origin/btopk64
rm -rf build *.so && MAX_JOBS=192 python setup.py build_ext --inplace 2>&1 | tail -n 100
python test_dq_2sm_mma.py > bug.txt 2>&1