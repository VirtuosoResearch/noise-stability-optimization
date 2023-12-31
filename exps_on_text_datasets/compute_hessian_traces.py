import argparse
import collections
import os
import random

import datasets
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, load_metric

import transformers
from data_loader.load_data_fns import load_glue_tasks
from parse_config import ConfigParser
from trainer import *
from transformers import (AdamW, AutoModelForSequenceClassification,
                          SchedulerType, get_scheduler)
from utils import deep_copy, prepare_device
from utils.hessian import compute_hessian_traces, set_seed, compute_eigenvalue
from utils.util import prepare_inputs


def main(config, args):
    set_seed(0)
    logger = config.get_logger('train')
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    
    # Load dataset
    train_data_loader, valid_data_loader, test_data_loader, transformers_config = load_glue_tasks(
        args.task_name, logger=logger,
        model_name_or_path=args.model_name_or_path,
        pad_to_max_length=config["data_loader"]["args"]["pad_to_max_length"],
        max_length=config["data_loader"]["args"]["max_length"],
        train_batch_size=config["data_loader"]["args"]["train_batch_size"],
        eval_batch_size=config["data_loader"]["args"]["eval_batch_size"]
    )
    test_data_loader = None

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)

    # Load pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=transformers_config,
    )

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    file = os.path.join("./saved/", args.checkpoint_dir)
    model.load_state_dict(
            torch.load(os.path.join(file, f"model_epoch_{args.epoch}.pth"))["state_dict"]
        )
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    # trace_dir = "./traces/{}_{}_layer_traces.npy".format(
    #     args.task_name, args.save_name
    # )
    
    max_traces = []
    sample_count = 0
    
    hessian_traces = []
    hessian_lambdas = []
    model.eval()
    for _, batch in enumerate(train_data_loader):
        batch = prepare_inputs(batch, device)
        outputs = model(**batch)
        loss = outputs.loss

        layer_traces, _ = compute_hessian_traces(model, loss, device = device)
        lambda_1, _ = compute_eigenvalue(model, loss, device=device, top_n=1) 

        hessian_traces.append(layer_traces)
        hessian_lambdas.append(np.array(lambda_1[0]))

        print(np.mean(np.array(hessian_traces), axis=0), np.mean(np.array(hessian_traces), axis=0).sum())
        print(np.mean(np.array(hessian_lambdas), axis=0), np.mean(np.array(hessian_lambdas), axis=0).sum())
        logger.info("========== Batch Complete ==========")
        # if sample_count == 0:
        #     max_traces = layer_traces
        # else:
        #     max_traces = np.maximum(max_traces, layer_traces)
        # logger.info("Current layer traces: {}".format(layer_traces))
        # logger.info("Traces: {}".format(max_traces))
        # logger.info("Traces sum: {}".format(np.sum(max_traces)))
        # np.save(trace_dir, max_traces)

        sample_count += 1
        if sample_count > args.sample_size:
            break
    print("Sum of trace: {}".format(np.mean(np.array(hessian_traces), axis=0).sum()))
    print("Sum of top-1 eigenvalues: {}".format(np.mean(np.array(hessian_lambdas), axis=0).sum()))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    args.add_argument('--task_name', type=str, default="mrpc")
    args.add_argument('--model_name_or_path', type=str, default="bert-base-uncased")
    args.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    args.add_argument("--epoch", type=int, default=1)

    args.add_argument("--checkpoint_dir", type=str, default="mrpc_False")
    args.add_argument("--checkpoint_name", type=str, default="model_best")
    args.add_argument("--save_name", type=str, default="finetuned_train")
    args.add_argument("--sample_size", type=int, default=100)
    args.add_argument("--early_stop", type=int, default=20)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--epochs'], type=int, target='trainer;num_train_epochs'),
        CustomArgs(['--warm_up'], type=int, target='trainer;num_warmup_steps'),
    ]
    config, args = ConfigParser.from_args(args, options)
    main(config, args)
