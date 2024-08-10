import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time
import datetime
import json

data_path = "/home/dmls/amirmazlaghani/fashion_mnist_data"

train_batch_size = 128
test_batch_size = 128
learning_rate = 0.001
num_epochs = 10

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
        return self.fc3(out)

def setup(backend):
    dist.init_process_group(backend=backend)

def find_free_port():
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def load_data(rank, world_size):
    train_set = torchvision.datasets.FashionMNIST(data_path, download=True, train=True, transform=transforms.Compose([transforms.ToTensor()]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, sampler=train_sampler, batch_size=train_batch_size, shuffle=False, persistent_workers=True, num_workers=1, pin_memory=True)

    test_set = torchvision.datasets.FashionMNIST(data_path, download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, num_replicas=world_size, rank=rank)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, sampler=test_sampler, batch_size=test_batch_size, shuffle=False, persistent_workers=True, num_workers=1, pin_memory=True)

    return train_loader, test_loader

def train(rank, world_size, master_port, backend, timeout):
    setup(backend)
    torch.cuda.set_device(rank)
    train_loader, test_loader = load_data(rank, world_size)

    model = FashionMNISTCNN().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=0.9)

    start_epoch = 0
    snapshot_path = 'training_snapshot.json'
    if os.path.exists(snapshot_path):
        with open(snapshot_path, 'r') as file:
            snapshot = json.load(file)
            start_epoch = snapshot['epoch']
            model.load_state_dict(torch.load(snapshot['model_state']))
            optimizer.load_state_dict(torch.load(snapshot['optimizer_state']))

    start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
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
    print(f"Rank: {rank}, Training Time: {end_time - start_time} seconds")

    if rank == 0:
        torch.save(model.state_dict(), 'model_final.pt')
        torch.save(optimizer.state_dict(), 'optimizer_final.pt')
        with open('final_snapshot.json', 'w') as file:
            json.dump({'epoch': num_epochs, 'model_state': 'model_final.pt', 'optimizer_state': 'optimizer_final.pt'}, file)

    if rank == 0:
        print("Finished Training")

if __name__ == "__main__":
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = torch.cuda.device_count()
    master_port = find_free_port()
    backend = 'nccl'
    timeout = datetime.timedelta(seconds=10)
    train(rank, world_size, master_port, backend, timeout)

