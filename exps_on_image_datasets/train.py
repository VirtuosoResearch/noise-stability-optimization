import os
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import *
from utils import prepare_device, deep_copy
from data_loader.random_noise import label_noise

from model.modeling_vit import VisionTransformer, CONFIGS
from utils.dual_t import get_transition_matrices, compose_T_matrices
from utils.sam import SAM
from utils.nso import NSO

'''
TODO:
    - Add the algorithms
    - Test which datasets work
'''

def main(config, args):
    logger = config.get_logger('train')

    # setup data_loader instances
    if config["data_loader"]["type"] == "CaltechDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, idx_start = 0, img_num = 30, phase = "train")
        valid_data_loader = config.init_obj('data_loader', module_data, idx_start = 30, img_num = 20, phase = "val")
        test_data_loader = config.init_obj('data_loader', module_data, idx_start = 50, img_num = 20, phase = "test")
    elif config["data_loader"]["type"] == "AircraftsDataLoader" \
        or config["data_loader"]["type"] == "DomainNetDataLoader" \
        or config["data_loader"]["type"] == "CXRDataLoader" :
        train_data_loader = config.init_obj('data_loader', module_data, phase = "train")
        valid_data_loader = config.init_obj('data_loader', module_data, phase = "val")
        test_data_loader = config.init_obj('data_loader', module_data, phase = "test")
        # valid_data_loader = test_data_loader
    elif config["data_loader"]["type"] == "BirdsDataLoader" or \
        config["data_loader"]["type"] == "CarsDataLoader" or \
        config["data_loader"]["type"] == "DogsDataLoader" or \
        config["data_loader"]["type"] == "IndoorDataLoader" or \
        config["data_loader"]["type"] == "Cifar10DataLoader" or \
        config["data_loader"]["type"] == "Cifar100DataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, valid_split = 0.1, phase = "train")
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = config.init_obj('data_loader', module_data, phase = "test")
    elif config["data_loader"]["type"] == "FlowerDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data)
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = train_data_loader.split_test()
    elif config["data_loader"]["type"] == "AnimalAttributesDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data)
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = train_data_loader.split_test()
    elif config["data_loader"]["type"] == "MessidorDataLoader" or \
        config["data_loader"]["type"] == "AptosDataLoader" or \
        config["data_loader"]["type"] == "JinchiDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, valid_split = 0.2, test_split=0.2, phase = "train")
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = train_data_loader.split_test()

    if args.synthetic_noise:
        if config["data_loader"]["type"] == "DomainNetDataLoader" or config["data_loader"]["type"] == "AnimalAttributesDataLoader":
            train_data_loader.dataset.labels = train_data_loader.dataset.true_labels
        wrong_indices, train_labels_old = label_noise(
            train_data_loader.dataset, train_data_loader.sampler.indices, args.noise_rate, symmetric=True
        )
        logger.info("Randomizing {} number of labels".format(wrong_indices.shape[0]))
    else:
        if config["data_loader"]["type"] == "DomainNetDataLoader" or config["data_loader"]["type"] == "AnimalAttributesDataLoader":
            train_labels_old = train_data_loader.dataset.true_labels[train_data_loader.sampler.indices]
        else:
            train_labels_old = None

    # If small data, shrink training data size
    assert 0 < args.data_frac <= 1
    if args.data_frac < 1:
        train_data_len = len(train_data_loader.sampler)
        train_data_loader.sampler.indices = train_data_loader.sampler.indices[:int(train_data_len*args.data_frac)]
        train_labels_old = None # 
        if args.downsample_test:
            test_data_len = len(test_data_loader.sampler)
            val_data_len = len(valid_data_loader.sampler)
            test_data_loader.sampler.indices = test_data_loader.sampler.indices[:int(test_data_len*args.data_frac)]
            valid_data_loader.sampler.indices = valid_data_loader.sampler.indices[:int(val_data_len*args.data_frac)]
    logger.info("Train Size: {} Valid Size: {} Test Size: {}".format(
        len(train_data_loader.sampler), 
        len(valid_data_loader.sampler), 
        len(test_data_loader.sampler)))

    # build model architecture, then print to console
    if args.is_vit:
        vit_config = CONFIGS[args.vit_type]
        model = config.init_obj('arch', module_arch, config = vit_config, img_size = args.img_size, zero_head=True)
        model.load_from(np.load(args.vit_pretrained_dir))
    else:
        model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    source_state_dict = deep_copy(model.state_dict())
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    accuracies = []
    
    if 'domain' in config['data_loader']['args']:
        domain_name = config['data_loader']['args']['domain']
    else:
        domain_name = None
    
    for run in range(args.runs):
        model.reset_parameters(source_state_dict)
        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        if args.train_sam:
            checkpoint_dir = os.path.join(
            "./saved_hessians", 
            "{}_{}_sam_{}_ada_{}".format(config["arch"]["type"], config["data_loader"]["type"], args.sam_rho, args.sam_adaptive))
            base_optimizer = getattr(torch.optim, config["optimizer"]["type"])
            optimizer = SAM(model.parameters(), base_optimizer, rho=args.sam_rho, 
                            adaptive=args.sam_adaptive, **dict(config["optimizer"]["args"]))
            lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer.base_optimizer)
            trainer = SAMTrainer(model, criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            lr_scheduler=lr_scheduler,
                            checkpoint_dir=checkpoint_dir)
        elif args.train_nso:
            checkpoint_dir = os.path.join(
                "./saved_hessians",
                "{}_{}_nso_{}_{}_{}_{}".format(config["arch"]["type"], config["data_loader"]["type"], args.nso_lam, args.nso_sigma, args.num_perturbs, args.use_neg))
            base_optimizer = getattr(torch.optim, config["optimizer"]["type"])
            optimizer = NSO(model.parameters(), base_optimizer, sigma=args.nso_sigma, **dict(config["optimizer"]["args"]))
            lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer.base_optimizer)
            trainer = NSOTrainer(model, criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            lr_scheduler=lr_scheduler,
                            checkpoint_dir=checkpoint_dir,
                            nso_lam=args.nso_lam,
                            num_perturbs=args.num_perturbs,
                            use_neg=args.use_neg)
        else:
            checkpoint_dir = os.path.join(
            "./saved_hessians", 
            "{}_{}_{}_{}_{:.4f}_{:.4f}_run_{}".format(config["arch"]["type"], config["data_loader"]["type"], 
                                    config["reg_method"], config["reg_norm"],
                                    config["reg_extractor"], config["reg_predictor"], run))
            checkpoint_dir = checkpoint_dir + ("_synthethic_{}".format(args.noise_rate) if args.synthetic_noise else "")
            trainer = ConstraintTrainer(model, criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            lr_scheduler=lr_scheduler,
                            checkpoint_dir = checkpoint_dir,
                            gradient_update_step=args.gradient_update_step)
                        

        trainer.train()
        accuracies.append(trainer.test())
    logger.info("Test Accuracy {:1.4f} +/- {:1.4f}".format(np.mean(accuracies), np.std(accuracies)))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, nargs='+',
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--runs', type=int, default=3)
    args.add_argument('--data_frac', type=float, default=1.0)
    args.add_argument('--downsample_test', action="store_true")
    
    args.add_argument('--is_vit', action="store_true")
    args.add_argument('--img_size', type=int, default=224)
    args.add_argument("--vit_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    args.add_argument("--vit_pretrained_dir", type=str, default="checkpoints/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")

    args.add_argument('--synthetic_noise', action="store_true")
    args.add_argument('--noise_rate', type=float, default=0.0)

    args.add_argument('--constraint_reweight', action="store_true")
    args.add_argument('--load_matrix', action="store_true")
    args.add_argument('--reweight_epoch', type=int, default=1)
    args.add_argument('--reweight_noise_rate', type=float, default=0.8)

    args.add_argument('--gradient_update_step', type=int, default=1)

    args.add_argument('--train_sam', action="store_true")
    args.add_argument('--sam_rho', type=float, default=0.05)
    args.add_argument('--sam_adaptive', action="store_true")

    args.add_argument('--train_nso', action="store_true")
    args.add_argument('--use_neg', action="store_true")
    args.add_argument('--nso_sigma', type=float, default=0.01)
    args.add_argument('--num_perturbs', type=int, default=1)
    args.add_argument('--nso_lam', type=float, default=0.5)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--model'], type=str, target="arch;type"),
        CustomArgs(['--weight_decay'], type=float, target="optimizer;args;weight_decay"),
        CustomArgs(['--reg_method'], type=str, target='reg_method'),
        CustomArgs(['--reg_norm'], type=str, target='reg_norm'),
        CustomArgs(['--reg_extractor'], type=float, target='reg_extractor'),
        CustomArgs(['--reg_predictor'], type=float, target='reg_predictor'),
        CustomArgs(['--scale_factor'], type=float, target="scale_factor"),
        CustomArgs(['--domain'], type=str, target="data_loader;args;domain"),
        CustomArgs(['--sample'], type=int, target="data_loader;args;sample"),
        CustomArgs(['--early_stop'], type=int, target="trainer;early_stop"),
        CustomArgs(['--epochs'], type=int, target="trainer;epochs"),
    ]
    config, args = ConfigParser.from_args(args, options)
    print(config)
    main(config, args)
