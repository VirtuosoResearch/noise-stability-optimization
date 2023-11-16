### Training GCN Models on Graph Prediction Datasets

**Training a GCN model on a graph prediction dataset**

Use `train_graph_pred.py` to train a GCN model. We use the datasets from TUDatasets. For example, use the following script to train a GCN model on the COLLAB dataset for 200 epochs. The dataset will be loaded automatically.

```script
python train_graph_pred.py --dataset COLLAB --num_layers 2 --device 0 --runs 1 --hidden 64 --epochs 200
```

**Evaluating the noise stability and Hessian-based measures on GCN**

Use the following scripts to compute noise stability and Hessian-based measures. We use Hessian vector multiplication tools from PyHessian (Yao et al., 2020).

- `compute_noise_stability.py` computes the averaged noise stability with a given perturbation scale $\sigma$
- `compute_hessian_traces.py` computes the trace of loss's Hessian of each layer in a neural network. 

Follow the bash script examples to run the commands. 

```bash
# Make sure you have the checkpoint saved in the ./saved/ folder
# The checkpoint will be automatically saved if one runs the training code above
# One can also modify the folder name inside the scrips in Line 170 and Line 100
python compute_noise_stability.py --dataset COLLAB --model gcn --num_layers 2 --hidden 64 --jk_type last --device 1 --fold_idx 0 --run 1

python compute_hessian_trace.py --dataset COLLAB --model gin --num_layers 4 --hidden 64 --jk_type last --aggr mean --device 1 --fold_idx 0 --run 1
```

