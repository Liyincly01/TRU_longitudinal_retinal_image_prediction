# TRU_longitudinal_faf

Reference implementation for *"Training-inference input alignment outweights framework choice in longitudinal retinal image prediction"* (in submission).


## Install

```bash
pip install -r requirements.txt
```
Install PyTorch separately per your CUDA version (see [pytorch.org](https://pytorch.org)).

## Usage

```bash
# Train (example: TRU)
python train.py --method tru --data_dir PATH --output_dir runs/tru

# Five-configuration benchmark
python evaluate.py --output_dir eval_results \
    --faf_holdout_dir PATH \
    --ckpt_std_ddim      runs/std_ddim/checkpoint_best.pt \
    --ckpt_ia_nonlinear  runs/ia_nonlinear/checkpoint_best.pt \
    --ckpt_ia_linear     runs/ia_linear/checkpoint_best.pt \
    --ckpt_tru           runs/tru/checkpoint_best.pt

# Supplementary analyses
python analysis/task_entropy.py       --data_dir PATH --output_dir OUT
python analysis/posterior_collapse.py --data_dir PATH --output_dir OUT \
    --ckpt_ia_nonlinear CKPT --ckpt_ia_linear CKPT

# 2D-BrLP benchmark
python external/train_brlp_2d.py --data_dir PATH --output_dir runs/brlp_2d
```

See [docs/data_format.md](docs/data_format.md) for the expected filename schema.

## Data and weights

No data or trained weights are shipped. Model weights are available on request to the corresponding author (TODO(author): add contact email).

## Citation

TODO(author): add arXiv / DOI and BibTeX once assigned.

## License

MIT — see [LICENSE](LICENSE). The 2D-BrLP adaptation in `external/` follows Puglisi et al., *Medical Image Analysis* 2025; please cite the original if you use that component.
