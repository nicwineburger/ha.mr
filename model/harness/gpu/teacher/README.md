# g-teacher768 weights (training-time artifact)

The 71.7M-parameter distillation teacher from the GPU campaign
(dim 768, 12 layers, 12 heads, mlp 2304, vocab-1024, ctx 96;
1.5693 bits/char on the shared eval set - see ../RESULTS.md).

This model **never ships**: it exists to provide soft targets for
distilling future small candidates, and is archived here so it
survives the Modal volume. It is stored in the standard
hamr-url-model-v2 format (f16, token manifest in header) but with
`linkVersion: 0` - it is NOT part of the payload version sequence
and must never be deployed as a decode model.

Split into <100MB parts for GitHub. Reassemble with:

```sh
cat url-model-teacher768.bin.part00 url-model-teacher768.bin.part01 \
  > url-model-teacher768.bin
sha256sum -c SHA256SUMS
```

To reuse it as a distillation teacher on Modal, upload the
reassembled file to the volume, or skip it entirely if
`ckpt-live-g-teacher768.pt` (fp32, with optimizer state) is still on
the `hamr-gpu` volume - the harness's `distill_from` mode loads that
checkpoint directly. To rebuild a torch state dict from this file,
follow the tensor mapping in `eval_shipped()` in ../modal_train.py.
