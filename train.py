import argparse

import torch
import torch.utils.data
import torchvision.models as models
import archml.utils
import archml.utils.distributed as dist
import functional as f


def initialize_parser():
    parser = argparse.ArgumentParser('', add_help=False)

    # required argument
    parser.add_argument('data', default=None, type=str, help='the path to dataset')

    # arguments
    parser.add_argument('--device', default='cuda', type=str, help='the device to use, disabled at distributed mode')
    parser.add_argument('--epochs', default=300, type=int, help='the number of total iterations to run (default: 300)')
    parser.add_argument('--resume', default=None, type=str, help='resume training from the checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, help='the manual epoch number (useful on restarts)')
    parser.add_argument('--seed', default=None, type=int, help='the seed for reproducibility')
    parser.add_argument('--save-dir', default=None, type=str, help='the checkpoint will be saved here')
    parser.add_argument('--tensorboard-dir', default=None, type=str, help='Tensorboard log will be saved here')

    # arguments for model
    parser.add_argument('--num-classes', default=1000, type=int, help='the number of classes in the dataset')

    # arguments for dataloader
    parser.add_argument('--batch-size', default=64, type=int, help='mini batch size for each device (default: 64)')
    parser.add_argument('--workers', default=8, type=int, help='the number of dataloader workers (default: 8)')
    parser.add_argument('--pin-memory', action='store_true', help='pin memory for more efficient data transfer')
    parser.add_argument('--no-pin-memory', action='store_false', dest='pin-memory')
    parser.set_defaults(pin_memory=True)

    # arguments for optimizer
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')

    # arguments for learning scheduler
    parser.add_argument('--step-size', default=30, type=int, help='')
    parser.add_argument('--gamma', default=1e-1, type=float, help='')

    # arguments for distributed data parallel mode
    # parser.add_argument('--world-size', default=1, type=int, help='the number of nodes for distributed mode')
    # parser.add_argument('--dist-url', default='env://', type=str, help='the url for distributed mode')

    return parser.parse_args()


def main(args):
    dist.initialize_distributed_mode(args)

    device = torch.device(args.device)

    model = models.resnet50()
    model.to(device)
    model_non_distributed = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpus])
        model_non_distributed = model.module

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = torch.nn.CrossEntropyLoss()
    evaluator = archml.nemo.Accuracy(topk=(1, 5))

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_non_distributed.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        scheduler.step(args.start_epoch)

    for epoch in range(args.start_epoch, args.epochs):
        mean_loss_train, mean_score_train = f.train(..., model, optimizer, criterion, evaluator, epoch, args)
        mean_loss_valid, mean_score_valid = f.valid(..., model, criterion, evaluator, epoch, args)


if __name__ == '__main__':
    config = initialize_parser()
    archml.utils.enable_reproducibility(config.seed)
    main(config)
