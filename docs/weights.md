# TRU checkpoint

The inference asset is `tru_ema_mh10.pt`, accompanied by `tru_ema_mh10.json` and `SHA256SUMS`.

Download the assets from [Releases](https://github.com/Liyincly01/TRU_longitudinal_retinal_image_prediction/releases).

| Setting | Value |
| --- | --- |
| Architecture | Temporal Retinal U-Net, 22,827,268 parameters |
| Weights | EMA |
| Image size | 256×256, grayscale |
| Training history limit | 10 visits |
| Configured training schedule | 150 epochs |
| Saved checkpoint | End of epoch 90 |

The file contains EMA tensors, training configuration and the saved epoch index. It omits optimizer state, patient paths and training logs. The accompanying JSON records the source and exported-file SHA256 checksums.

```bash
sha256sum -c SHA256SUMS
python evaluate.py --manifest private/visits.csv --cohort optos --split test \
  --ckpt_tru weights/tru_ema_mh10.pt --output_dir outputs/tru
```

Inference defaults to the recorded ten-visit limit. Add `--max_history N` to use a different number of recent visits; this changes the inference history without changing the checkpoint. Fewer available visits are padded and masked. Clinical endpoint evaluation additionally requires atrophy segmentation weights and registered foveal coordinates.
