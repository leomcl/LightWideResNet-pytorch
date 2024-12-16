import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import os
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
from google.colab import drive
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Mount Google Drive
drive.mount('/content/drive')

# Create checkpoint directory in Google Drive
checkpoint_dir = '/content/drive/MyDrive/CIFAR100_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using GPU:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

cudnn.benchmark = True  # Enable CUDNN autotuner

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=16, num_classes=100, widen_factor=8, dropRate=0.25):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        with torch.cuda.amp.autocast():
            out = self.conv1(x)
            out = F.dropout(out, p=0.1, training=self.training)
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, 8)
            out = out.view(-1, self.nChannels)
            return self.fc(out)

def data_loader(data_dir,
                batch_size,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False):
    
    # Mixed Precision Training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomErasing(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Simpler transform for testing/validation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    if test:
        dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False,
            download=True, transform=transform_test
        )

        data_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=2
        )

        return data_loader

    # Load training dataset
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True,
        download=True, transform=transform_train
    )

    valid_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True,
        download=True, transform=transform_test
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=6,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True,
        drop_last=True
    )
 
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        sampler=valid_sampler,
        num_workers=4,
        pin_memory=True
    )

    return (train_loader, valid_loader)

def train_model():
    # Create directories for saving data
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'/content/drive/MyDrive/CIFAR100_results_{run_id}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize tracking lists
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    learning_rates = []
    epoch_times = []
    
    # Load data
    batch_size = 512
    train_loader, valid_loader = data_loader(data_dir='./data', batch_size=batch_size)
    test_loader = data_loader(data_dir='./data', batch_size=batch_size, test=True)
    
    # Initialize model
    model = WideResNet(depth=16, num_classes=100, widen_factor=8, dropRate=0.25).to(device)
    
    
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.15,
        reduction='mean',
        weight=torch.ones(100).to(device)
    ).to(device)
    initial_lr = 0.2
    optimizer = optim.SGD(
        model.parameters(),
        lr=initial_lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )
    
    # Add gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.3,
        epochs=85,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=25,
        final_div_factor=1e3,
        anneal_strategy='linear'
    )
    
    # Training loop
    num_epochs = 100
    best_accuracy = 0.0
    
    # Add mixed precision scaler
    scaler = GradScaler()
    
   
    accumulation_steps = 2
    effective_batch_size = batch_size * accumulation_steps
    
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        epoch_loss = 0.0
        
        model.train()
        running_loss = 0.0
        total_train = 0  # Initialize counter
        correct_train = 0  # Initialize counter
        optimizer.zero_grad()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels) / accumulation_steps
            
            # Gradient accumulation
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % 20 == 19:
                print(f'[Batch {i + 1}] Loss: {running_loss / 20:.3f}, '
                      f'Training Accuracy: {100 * correct_train / total_train:.2f}%')
                running_loss = 0.0
            
            if i % 50 == 0:
                torch.cuda.empty_cache()
        
        # Calculate epoch metrics
        epoch_loss = epoch_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        class_correct = [0] * 100
        class_total = [0] * 100
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Per-class accuracy
                c = (predicted == labels).squeeze()
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        accuracy = 100 * correct / total
        print(f"\nTest Accuracy: {accuracy:.2f}%")
        
        # Record metrics
        train_losses.append(epoch_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(accuracy)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        epoch_times.append((datetime.now() - epoch_start_time).total_seconds())
        
        # Save checkpoint
        try:
            checkpoint_name = f'WideResNet_epoch_{epoch+1:03d}_acc_{accuracy:.2f}.pth'
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'loss': loss.item()
            }
            
            torch.save(checkpoint, checkpoint_path)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_path = os.path.join(checkpoint_dir, 'WideResNet_best_model.pth')
                torch.save(checkpoint, best_model_path)
                print(f"New best model saved with accuracy: {accuracy:.2f}%")
                
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr}')
        
        # Create and save plots every 5 epochs
        if (epoch + 1) % 5 == 0:
            # Loss plot
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Training Loss')
            plt.title('Training Loss Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f'{results_dir}/loss_plot.png')
            plt.close()
            
            # Accuracy plot
            plt.figure(figsize=(10, 5))
            plt.plot(train_accuracies, label='Training Accuracy')
            plt.plot(test_accuracies, label='Test Accuracy')
            plt.title('Model Accuracy Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.savefig(f'{results_dir}/accuracy_plot.png')
            plt.close()
            
            # Learning rate plot
            plt.figure(figsize=(10, 5))
            plt.plot(learning_rates)
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.savefig(f'{results_dir}/lr_plot.png')
            plt.close()
            
            # Save metrics to CSV           
            metrics_df = pd.DataFrame({
                'epoch': range(1, epoch + 2),
                'train_loss': train_losses,
                'train_accuracy': train_accuracies,
                'test_accuracy': test_accuracies,
                'learning_rate': learning_rates,
                'epoch_time': epoch_times
            })
            metrics_df.to_csv(f'{results_dir}/training_metrics.csv', index=False)
            
            # Save per-class accuracies
            class_accuracies = [
                (100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0)
                for i in range(100)
            ]
            plt.figure(figsize=(15, 5))
            plt.bar(range(100), class_accuracies)
            plt.title('Per-Class Accuracy')
            plt.xlabel('Class Index')
            plt.ylabel('Accuracy (%)')
            plt.savefig(f'{results_dir}/class_accuracy_plot.png')
            plt.close()
    
    # Final summary
    summary = {
        'best_accuracy': best_accuracy,
        'total_training_time': sum(epoch_times) / 3600,  # in hours
        'average_epoch_time': sum(epoch_times) / len(epoch_times),
        'final_train_accuracy': train_accuracies[-1],
        'final_test_accuracy': test_accuracies[-1]
    }
    
    with open(f'{results_dir}/training_summary.txt', 'w') as f:
        for key, value in summary.items():
            f.write(f'{key}: {value}\n')

if __name__ == '__main__':
    train_model() 