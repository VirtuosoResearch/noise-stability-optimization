#### **Fine-tuning BERT on Text Classification Tasks**

**Fine-tuning BERT model on a text classification dataset with NSO**

Use `train_glue.py` to run the experiments of fine-tuning BERT-Base. Follow the bash script example to run the command. 

```bash
python train_glue_label_noise.py --config configs/config_glue.json --task_name mrpc \
    --epochs 5 --runs 1 --device 0 --model_name_or_path bert-base-uncased\
    --train_nsm --use_neg --nsm_lam 0 --num_perturbs 1 --nsm_sigma 0.01
```

**Evaluating the noise stability and Hessian-based measures on BERT**

Use the following scripts to compute noise stability and Hessian-based measures. We use Hessian vector multiplication tools from PyHessian (Yao et al., 2020).

- `compute_noise_stability.py` computes the averaged noise stability with a given perturbation scale $\sigma$
- `compute_hessian_traces.py` computes the trace of loss's Hessian of each layer in a neural network. 

Follow the bash script examples to run the commands. 

```bash
python compute_noise_stability.py --config configs/config_glue.json --task_name mrpc --device 0 \
--checkpoint_dir $specify_a_checkpoint_dir --checkpoint_name $specify_a_checkpoint_name --sample_size 100 --eps $specify_a_noise_scale # such as 0.01

python compute_hessian_traces.py --config configs/config_glue.json --task_name mrpc --device 0 \
    --checkpoint_dir $specify_a_checkpoint_dir --checkpoint_name $specify_a_checkpoint_name --save_name $specify_a_save_filename --sample_size 10
```

