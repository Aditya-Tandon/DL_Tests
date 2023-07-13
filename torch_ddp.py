import os
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed
import torch.utils.data
from apex import DistributedDataParallel as DDP
from apex import amp


class test_net(nn.Module):
    def __init__(self, num_classes=10):
        super(test_net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out).reshape(out.size(0), -1)
        out = self.fc_layer(out)
        return out


def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank,
    )

    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    model = test_net().cuda(gpu)

    batch_size = 100
    train_dataset = torchvision.datasets.MNIST(
        root="/kaggle/working/mnist_data",
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )

    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e4)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    # model = DDP(model)

    total_steps = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            y = model.forward(images)
            loss = loss_fn(y, labels)
            optimizer.zero_grad()
            loss.backward()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0 and gpu == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, args.epochs, i + 1, total_steps, loss.item()
                    )
                )


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("-n", "--nodes", default=1, type=int, metavar="N")
    args_parser.add_argument(
        "-g", "--gpus", default=1, type=int, help="number of GPUs"
    )
    args_parser.add_argument(
        "-nr", "--nr", default=0, type=int, help="ranking within the nodes"
    )
    args_parser.add_argument(
        "-e",
        "--epochs",
        default=0,
        type=int,
        help="number of epochs",
    )
    args = args_parser.parse_args()
    args.world_size = args.nodes * args.gpus
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "8888"
    mp.spawn(train, nprocs=args.gpus, args=(args,))
    # train(0, gpu)


if __name__ == "__main__":
    main()
