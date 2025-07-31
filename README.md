# tBN-CSDI

This is a modified version of CSDI to add a blue noise schdule to the standard noise process. I added a lot of comments to help identify the new changes

The only file that was modified was main_model.py in the CSDI folder. 

Modifications:
1. gen_bn.py is used to precompute blue noise for the main process to use. This is not fully refined as the paper does not explain exactly how they did this part.
   
2. The next modification is to use a rectified mapping for the noise. The original paper mentioned using a flow based matching however this lead to major overfitting so I went with a mapping based off hungarian assignment which performed much better

3. The blending of Gaussian noise and blue noise is time dependent based on a gamma function which is added to calculate a blending factor based on the time step sampled.

4. The foward and reverse process where modified to combine the Gaussian noise to the 2D blue noise based off of the gamma blending function. they were also modified to apply the rectified mapping to the blended noise. The rest of the foward and reverse process where left the same.

5. there was no modification to any other parts of the model so the effects on the performance could be directly associated to the addition of the blue noise scheduler. 

6. I also was playing around with using a precomputed Feature X Time covariance decay matrix to sample from instead of standard blue noise, This worked quite well and is in main_model_cov.py



To Download the Physio Data:

python download.py physio

This will:
- Download the dataset from PhysioNet
- Extract it to `data/physio/set-a/`
- Create the necessary directory structure

### 2. Precompute Covariance Matrix

The model uses a precomputed 64x64 covariance matrix for correlated noise generation. This will be automatically created on first run, but you can precompute it manually:

```python
import torch
from main_model import CSDI_base

# Create a dummy config for matrix computation
config = {
    "model": {
        "rho_feat": 0.5,
        "rho_time": 0.5,
        "cov_save_path": "cov_matrix_tile.pt"
    }
}

# This will create the covariance matrix file
model = CSDI_base(target_dim=35, config=config, device='cpu')
```

Or Just:
python gen_bn.py

The covariance matrix will be saved as `cov_matrix_tile.pt` in the root directory.

## Running Experiments

### Training and Testing

Train a new model and run imputation:

```bash
python exe_physio.py --testmissingratio 0.1 --nsample 100
```

Parameters:
- `--testmissingratio`: Missing data ratio (default: 0.1)
- `--nsample`: Number of samples for evaluation (default: 100)
- `--nfold`: Cross-validation fold (0-4, default: 0)
- `--device`: Device to use (default: 'cuda:0')
- `--seed`: Random seed (default: 1)

### Using Pretrained Model

If you have a pretrained model, you can use it for testing:

```bash
python exe_physio.py --modelfolder pretrained --testmissingratio 0.1 --nsample 100
```

### Unconditional Generation

For unconditional generation (without conditioning on observed data):

```bash
python exe_physio.py --unconditional --testmissingratio 0.1 --nsample 100
```

## Configuration

The model configuration is defined in `config/base.yaml`. Key parameters include:

- **Diffusion**: Number of steps, beta schedule, network architecture
- **Model**: Time/fature embedding dimensions, target strategy
- **Noise**: Correlation parameters (rho_feat, rho_time), gamma schedule

## Output

Results will be saved in:
- `save/physio_fold{N}_{timestamp}/` - Model checkpoints and results
- `cov_matrix_tile.pt` - Precomputed covariance matrix

## Visualization

Use the provided Jupyter notebook to visualize results:

```bash
jupyter notebook visualize_examples.ipynb
```
