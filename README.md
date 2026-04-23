# TRU_longitudinal_faf

Reference implementation for *"Training-inference input alignment outweighs framework choice in longitudinal retinal image prediction"* (in submission).


## Install

```bash
pip install -r requirements.txt
```
Install PyTorch separately per your CUDA version (see [pytorch.org](https://pytorch.org)).

## Usage

```bash
# Train (example: TRU)
python train.py --method tru --data_dir PATH --output_dir runs/tru

# Diagnostic analyses
python analysis/task_entropy.py       --data_dir PATH --output_dir OUT
python analysis/posterior_collapse.py --data_dir PATH --output_dir OUT \
    --ckpt_ia_nonlinear CKPT --ckpt_ia_linear CKPT

# 2D-BrLP benchmark
python external/train_brlp_2d.py --data_dir PATH --output_dir runs/brlp_2d

```
See [docs/data_format.md](docs/data_format.md) for the expected filename schema.


## Citation
 
https://doi.org/10.48550/arXiv.2604.16955

## License

MIT — see [LICENSE](LICENSE). The 2D-BrLP adaptation in `external/` follows Puglisi et al., *Medical Image Analysis* 2025; please cite the original if you use that component.
