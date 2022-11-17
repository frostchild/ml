from typing import Any, Tuple

import numpy as np
import torch.utils.data


def train(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    evaluator: torch.nn.Module,
    epoch: int,
    args,
) -> Tuple[Any, Any]:
    model.train()
    losses, scores = list(), list()
    for idx, batch in enumerate(dataloader):
        inputs, targets = batch
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        score = evaluator(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        scores.append(score.item())
        print(f'\rEpoch[{epoch}/{args.epochs}] - batch[{idx + 1}/{len(dataloader)}]', end='')
    return np.nanmean(losses), np.nanmean(scores)


@torch.no_grad()
def valid(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    evaluator: torch.nn.Module,
    epoch: int,
    args,
) -> Tuple[Any, Any]:
    model.eval()
    losses, scores = list(), list()
    for idx, batch in enumerate(dataloader):
        inputs, targets = batch
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        score = evaluator(outputs, targets)
        losses.append(loss.item())
        scores.append(score.item())
        print(f'\rEpoch[{epoch}/{args.epochs}] - batch[{idx + 1}/{len(dataloader)}]', end='')
    return np.nanmean(losses), np.nanmean(scores)
