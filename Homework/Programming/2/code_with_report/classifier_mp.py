import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import datetime
import os

data_path = "/home/dmls/amirmazlaghani/fashion_mnist_data"
train_batch_size = 32
test_batch_size = 32
learning_rate = 0.001
num_epochs = 10

def setup(rank, world_size, master_port, backend, timeout):
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, timeout=timeout)

def find_free_port():
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(in_features=128*3*3, out_features=1024)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

def load_data(rank, world_size):
    train_set = torchvision.datasets.FashionMNIST(data_path, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, sampler=train_sampler, batch_size=train_batch_size, shuffle=False, persistent_workers=True, num_workers=1, pin_memory=True)

    test_set = torchvision.datasets.FashionMNIST(data_path, download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, num_replicas=world_size, rank=rank)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, sampler=test_sampler, batch_size=test_batch_size, shuffle=False, persistent_workers=True, num_workers=1, pin_memory=True)

    return train_loader, test_loader

def train(rank, world_size, master_port, backend, timeout):
    setup(rank, world_size, master_port, backend, timeout)
    torch.cuda.set_device(rank)
    train_loader, test_loader = load_data(rank, world_size)

    model = FashionMNISTCNN().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=learning_rate)

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = ddp_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 99:
                print(f"Rank: {rank}, Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {loss.item():.3f}")

        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(rank), labels.to(rank)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Rank: {rank}, Epoch: {epoch + 1}, Accuracy: {accuracy:.2f}%")

    end_time = time.time()
    cuda_mem = torch.cuda.max_memory_allocated(device=rank) / (1024 ** 2)
    print(f"Rank: {rank}, Training Time: {end_time - start_time} seconds, Max Memory Allocated: {cuda_mem} MB")

    if rank == 0:
        print("Finished Training")

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    master_port = find_free_port()
    backend = 'nccl'
    timeout = datetime.timedelta(seconds=10)
    mp.spawn(train, nprocs=world_size, args=(world_size, master_port, backend, timeout), join=True)

