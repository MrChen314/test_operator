import torch
import pytest

from test_write_s_ds import run_write_s_ds


ROWS = 64
COLS = 64
HALF_COLS = 32
HALF_MASK = 0x0000FFFF
NEG_DELTA = -0.5
SM_SCALE = 0.125
EXPECTED_BAD_COLS = list(range(16, 32)) + list(range(48, 64))


def build_reference(device: torch.device, use_mask: bool, buggy_mask: bool):
    rows = torch.arange(ROWS, device=device, dtype=torch.float32)[:, None]
    cols = torch.arange(COLS, device=device, dtype=torch.float32)[None, :]

    logits = -3.5 + 0.03125 * cols + 0.01 * rows
    dp = -0.75 + 0.02 * cols - 0.003 * rows

    if use_mask:
        half_idx = torch.arange(HALF_COLS, device=device, dtype=torch.int64)
        fixed_half = ((torch.full_like(half_idx, HALF_MASK) >> half_idx) & 1).to(torch.bool)
        if buggy_mask:
            buggy_half = torch.cat(
                [fixed_half[: HALF_COLS // 2], torch.ones(HALF_COLS // 2, device=device, dtype=torch.bool)]
            )
            valid = torch.cat([buggy_half, buggy_half])[None, :]
        else:
            valid = torch.cat([fixed_half, fixed_half])[None, :]
        logits = torch.where(valid, logits, torch.full_like(logits, float("-inf")))

    s = torch.exp2(logits)
    ds = s * (dp + NEG_DELTA) * SM_SCALE
    return s.to(torch.bfloat16), ds.to(torch.bfloat16)


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def bad_columns(a: torch.Tensor, b: torch.Tensor):
    col_diff = (a.float() - b.float()).abs().amax(dim=0)
    return torch.nonzero(col_diff > 0, as_tuple=False).flatten().cpu().tolist()


def test_write_s_ds():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to run this test.")

    device = torch.device("cuda")
    (
        s_scalar,
        s_vector,
        s_fixed_mask,
        s_buggy_mask,
        ds_scalar,
        ds_vector,
        ds_fixed_mask,
        ds_buggy_mask,
    ) = run_write_s_ds()

    s_nomask_ref, ds_nomask_ref = build_reference(device, use_mask=False, buggy_mask=False)
    s_fixed_ref, ds_fixed_ref = build_reference(device, use_mask=True, buggy_mask=False)

    assert torch.equal(s_scalar, s_vector), "Scalar and vectorized S writes should match exactly."
    assert torch.equal(ds_scalar, ds_vector), "Scalar and vectorized dS writes should match exactly."

    assert max_diff(s_scalar, s_nomask_ref) < 1e-3, "Scalar S write deviates from reference."
    assert max_diff(ds_scalar, ds_nomask_ref) < 1e-3, "Scalar dS write deviates from reference."
    assert max_diff(s_fixed_mask, s_fixed_ref) < 1e-3, "Fixed-mask S write deviates from reference."
    assert max_diff(ds_fixed_mask, ds_fixed_ref) < 1e-3, "Fixed-mask dS write deviates from reference."

    s_bad_cols = bad_columns(s_buggy_mask, s_fixed_mask)
    ds_bad_cols = bad_columns(ds_buggy_mask, ds_fixed_mask)

    assert s_bad_cols == EXPECTED_BAD_COLS, f"Unexpected S mismatch columns: {s_bad_cols}"
    assert ds_bad_cols == EXPECTED_BAD_COLS, f"Unexpected dS mismatch columns: {ds_bad_cols}"

    assert torch.all(s_fixed_mask[:, 16:32] == 0)
    assert torch.all(s_fixed_mask[:, 48:64] == 0)
    assert torch.any(s_buggy_mask[:, 16:32] != 0)
    assert torch.any(s_buggy_mask[:, 48:64] != 0)

    print("Scalar and vectorized shared-memory writes match exactly.")
    print(f"Buggy-mask mismatches are isolated to columns: {s_bad_cols}")
    print("Conclusion: the 64daee regression is not the 128-bit SMEM address formula.")
    print("Conclusion: the regression comes from only masking the first 16 floats of each 32-float half-tile.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this test.")
    test_write_s_ds()
