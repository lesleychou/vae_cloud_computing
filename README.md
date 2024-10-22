## README
Project for cloud computing class.

### Getting Started

#### Installation

To create a suitable environment:
- `conda activate prvtel_env`
- `pip install -r requirements.txt`
- `pip uninstall rdt` (SDV installs rdt by default however we have included an added fix so this needs to be uninstalled to avoid conflicts)

#### GPU Support

This code has been tested both on CPU in the torch v1.9.0 given. But it has also been run on a GPU environment. The specifications for the device running this are as follows:

- NVIDIA GeForce RTX 3070 Laptop GPU
- CUDA v11.1
- cuDNN v8.1.0 for CUDA v11.1

### Usage

To understand how we preprocess data to prepare for the VAE input, check `utils.py`. 

To train a VAE model on cisco data (included in the repo) and generate synthetic data of it.
```python
cd e2e_system
python main.py
```
To tune the hyperparameter of VAE, check `e2e_system/config.py`. To tune the VAE structure, check `VAE.py`.

To evaluate the accuracy of generated synthetic data, run `e2e_system/Eval_query.py` in jupyter notebook.

### Your task and goal
Please tune the VAE model (feel free to change hyperparameters, VAE model structure, or/and data processing function), so that the query results can reach the accuracy we want in `e2e_system/Eval_query.py`. 



