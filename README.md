### Overview

We provide the implementation of the noise stability optimization algorithm in order to find flat minimizers. The algorithm adds random perturbations to the model weights and computes gradients from the perturbed weights to conduct gradient descent. We evaluate the Hessian trace and largest eigenvalue to validate the improved sharpness by our algorithm.

### Usage

The main implementation of the algorithm is the `NSO` optimizer as in `./exps_on_image_datasets/utils/nso.py`. We provide a simple code structure to use the algorithm in the following: 

```python
from utils.nso import NSO
from utils.bypass_bn import enable_running_stats, disable_running_stats
...

class args:
  # NSO parameters
  nsm_sigma = 0.01 # standard deviation sigma of the isotropic Gaussian distribution 
  nsm_perturbs = 1 # how many perturbations sampled (k in the algorithm), default: 1
  use_neg = True # if use two-point estimate, default: True
  nso_lam = 0 # weight of the unperturbed weight loss, default: 0
  

model = YourModel()
base_optimizer = torch.optim.SGD  # define an optimizer for the gradient descent update
optimizer = NSO(model.parameters(), base_optimizer, sigma=args.nso_sigma, **dict(config["optimizer"]["args"])) # pass in the sigma of sample distribution and other optimizer parameters, such as weight_decay
...

for input, output in data:
  
    # first forward-backward step: compute the gradients on the original weight (can be skipped if nso_lam == 0)
    enable_running_stats(model)

    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.store_gradients(zero_grad=True, store_weights=True, update_weight=self.nso_lam)

    # second forward-backward step: taking perturbations and computing gradients (main part)
    disable_running_stats(model)
    if args.num_perturbs != 0:
        update_weight = (1-args.nso_lam)/(2*args.num_perturbs) if args.use_neg else (1-args.nso_lam)/(args.num_perturbs)
        for i in range(self.num_perturbs):
            optimizer.first_step(zero_grad=True, store_perturb=True)
            criterion(model(data), target).backward()
            optimizer.store_gradients(zero_grad=True, store_weights=False, update_weight=update_weight)
            if args.use_neg:
                optimizer.first_step(zero_grad=True, store_perturb=False)
                criterion(model(data), target).backward()
                optimizer.store_gradients(zero_grad=True, store_weights=False, update_weight=update_weight)
    optimizer.second_step(zero_grad=True)
...
```

The full training logic is specified in the `NSOTrainer` from `./exps_on_image_datasets/trainer/nso_trainer.py`. 

### Experiments

We evaluate our algorithm through training networks across image, text, and graph datasets. 

**Requirements.** To install requirements:

```bash
pip install -r requirements.txt
```

**Fine-tuning on image classification datasets.** Enter the `./exps_on_image_datasets` folder to conduct fine-tuning experiments on image classification datasets. Please refer to `data/README.md` for the introductions to preparing data. 

- Use `train.py` to run experiments of fine-tuning ResNet/VisionTransformers on image datasets. 
- `compute_hessian_traces.py` computes the trace of loss's Hessian of each layer in a neural network. 
- `compute_noise_stability.py` computes the averaged noise stability with a given perturbation scale. 

**Examples.** We provide examples to run the scripts in the `exps_on_image_datasets/scripts` folder. 

Furthermore, we provide validation on text and graph classification datasets: 

- Enter `./exps_on_graph_datasets` folder to train and evaluate Hessian quantities on GCN networks. Please see the `README.md` inside the folder. 
- Enter `./exps_on_text_datasets` folder to train and evaluate Hessian quantities on BERT networks. Please see the `README.md` inside the folder. 

### Acknowledgment

Thanks to the authors of the following repositories for providing their implementation publicly available.

- **[SAM Optimizer (In PyTorch)](https://github.com/davda54/sam)**

- **[PyHessian](https://github.com/amirgholami/PyHessian)**
- **[ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)**
