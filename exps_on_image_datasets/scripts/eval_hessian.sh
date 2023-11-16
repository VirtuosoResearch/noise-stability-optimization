python compute_hessian_traces.py --config configs/config_constraint_cifar10.json \
    --model ResNet34 --batch_size 16 \
    --checkpoint_dir ResNet34_Cifar10DataLoader_nsm_0.0_0.01_2_True \
    --checkpoint_name model_best \
    --sample_size 1000 --device 1