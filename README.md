## Dream-Like Imagination with Large Embedding Models (LEMs)

We investigate whether large embedding models can “dream”: given heavily corrupted inputs, can they recover and recombine memories in embedding space and decode them back to coherent images?

**Setup.** We use a CLIP–unCLIP pipeline on Imagenette. Simple reconstruction nets (three-layer MLP and two Hopfield variants) learn to map corrupted inputs—either pixel-space crops/occlusions or Gaussian noise in CLIP embedding space—back to clean CLIP image embeddings. Decoding these embeddings with Stable unCLIP yields toy “memory recall” from severe fragments.

**Compositional blends.** We probe creativity by mixing reconstructed embeddings from two differently corrupted images (e.g., dog + fish, chainsaw + church) and decoding the blended latent. Coherent hybrids require successful recovery of each class before combination.

**Feature-space masking.** With frozen DINOv2 encoders, a transformer decoder is trained to inpaint masked features. We compare compressed vs. original representations on ImageNet-1k k-NN accuracy and qualitative “dream” visualizations.

**Takeaway.** Pattern completion in embedding space can underwrite both robust memory retrieval from fragments and compositional, dream-like recombination—offering a simple computational analogy to pre-linguistic thought and imagination.

## Repository Layout
- `src/`: Training and sampling code (CLIP reconstruction, Hopfield variants, unCLIP decoding, feature inpainting).
- `configs/`: OmegaConf configs for pretrained encoders/decoders and training recipes.
- `assets/`: Example images (e.g., pixabay cat).
- `notebooks/`: Analyses and demos (`demo.ipynb` for Colab-ready reconstruction; clip/unclip reconstruction notebook).
- `results/`: Local artifacts (legacy contents in `results/rae/` after reorg).
- `environment.yml`: micromamba/conda environment spec.

## Quickstart (local)
```bash
micromamba env create -f environment.yml
micromamba activate rae

# Download pretrained decoders/stats
pip install huggingface_hub
hf download nyu-visionx/RAE-collections --local-dir models

# Run a small reconstruction demo
micromamba run -n rae python notebooks/demo.ipynb  # or open in Jupyter/Colab
```

## Colab Demo
Open `notebooks/demo.ipynb` in Google Colab. It installs deps, clones the repo, downloads Imagenette and pretrained weights, trains the small MLP reconstructors, and visualizes reconstructions and blended “dream” scenes.

## Key Scripts
- `src/knn_restore.py`: CLIP reconstruction + mixing pipeline (MLP/Hopfield), Stable unCLIP decoding.
- `src/dataloader.py`: Imagenette loaders with crop/occlude/gaussian corruptions.
- `src/retrieval_imagination.py`: Fragment retrieval and imagination study.
- `src/eval_knn_pca.py`: Feature-space masking and k-NN evaluation for DINOv2 decoders.
