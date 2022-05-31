import torchvision.transforms as transforms
import torch
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from transformers.models.distilbert.modeling_distilbert import Embeddings
from torch import nn, Tensor
from typing import Optional, Union, Tuple, Dict, List, Any
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, PreTrainedModel, PreTrainedTokenizer)
import os
import numpy as np
from fi_code.utils import divide_to_batches, get_indices
from fi_code.fi import calculate_covariance, cholesky, flatten

"""
Image modality (CNN)
NN code is mostly taken from: https://github.com/NERSC/pytorch-examples/blob/main/models/resnet_cifar10.py
"""

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    @classmethod
    def ResNet18(cls):
        return cls(BasicBlock, [2, 2, 2, 2])

    @property
    def device(self):
        return next(self.parameters()).device

    def train_step(self, train_loader, criterion, optimizer):
        model = self
        device = model.device
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        batch_idx = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f'Train - Loss: {train_loss / (batch_idx + 1):.3f} | Acc: {correct / total:.3f}')

    def test_step(self, test_loader, criterion):
        model = self
        device = model.device
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            print(f'Test - Loss: {test_loss / (batch_idx + 1):.3f} | Acc: {correct / total:.3f}')

    def train_model(self, train_loader, test_loader, lr=0.005, epochs=15):
        model = self
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        for epoch in range(epochs):
            self.train_step(train_loader, criterion, optimizer)
            self.test_step(test_loader, criterion)
            scheduler.step()
        return model


def prepare_cifar10_data_loaders(batch_size=128, training_transforms: bool = True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train if training_transforms else transform_test)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False, num_workers=1)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader, test_loader, classes


def denormalize_cifar10_images(images: Tensor):
    mean = torch.tensor((0.4914, 0.4822, 0.4465))[:, None, None]
    std = torch.tensor((0.2023, 0.1994, 0.2010))[:, None, None]
    return images * std + mean


def calculate_image_covariances(train_loader, per_channel: bool = False):
    training_data, training_targets = get_indices(train_loader.dataset)
    covs, chols = [], []
    for label in range(len(training_targets.unique())):
        label_data = training_data[training_targets == label]
        label_data = flatten(label_data, 2)
        if per_channel:
            covariance, chol = [], []
            for i in range(label_data.shape[1]):
                covariance.append(calculate_covariance(label_data[:, i]))
                chol.append(cholesky(covariance[-1]))
            covariance, chol = torch.stack(covariance, dim=0), torch.stack(chol, dim=0)
        else:
            covariance = calculate_covariance(label_data.mean(dim=1))
            chol = cholesky(covariance)
        covs.append(covariance)
        chols.append(chol)
    return torch.stack(covs, dim=0), torch.stack(chols, dim=0)


"""
Text modality (Bi/LSTM) with embeddings of a pre-trained transformer model
"""

class LSTMClassifier(nn.Module):
    def __init__(self, embeddings: Embeddings, num_classes: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.1, bidirectional: bool = True):
        super(LSTMClassifier, self).__init__()
        self.embeddings = embeddings
        self.lstm = nn.LSTM(input_size=self.embeddings.embedding_dim, hidden_size=hidden_size,
                            batch_first=True, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        b = 2 if bidirectional else 1
        self.fc = nn.Linear(b * hidden_size, num_classes)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        m = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)
        embeddings = m.get_input_embeddings()
        return cls(embeddings, **kwargs)

    def get_input_embeddings(self):
        return self.embeddings

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        return torch.load(path)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids=None,
                attention_mask=None,
                inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        lstm_out, _ = self.lstm(inputs_embeds)
        x = lstm_out[:, -1]
        x = self.fc(x)
        return x

    def train_model(self,
                    train_dataloader: DataLoader,
                    test_dataloader: DataLoader,
                    lr: float = 1e-4,
                    n_epochs: int = 15):
        model = self
        device = model.device
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        for i in range(n_epochs):
            model.train()
            train_avg_loss = []
            test_avg_loss = []
            for idx, batch in enumerate(train_dataloader):
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                optimizer.zero_grad()
                scores = model(input_ids)
                loss = loss_function(scores, labels)
                train_avg_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            print(f'Train - Loss: {np.mean(train_avg_loss):.3f}')
            total = 0
            correct = 0
            with torch.no_grad():
                model.eval()
                for idx, batch in enumerate(test_dataloader):
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['label'].to(device)
                    scores = model(input_ids)
                    loss = loss_function(scores, labels)
                    preds = torch.argmax(scores, dim=1)
                    test_avg_loss.append(loss.item())
                    total += labels.shape[0]
                    correct += (preds == labels).sum().item()
            print(f'Test - Loss: {np.mean(test_avg_loss):.3f} | Acc: {correct / total:.3f}')
        return model


def create_preprocess_function(tokenizer, max_length: int = 32, padding_max_length: bool = True,
                               with_cls: bool = False):
    def preprocess_function(examples):
        text = tokenizer.batch_decode([t[-max_length-1:-1] for t in tokenizer(examples['text'])['input_ids']],
                                      skip_special_tokens=True, clean_up_tokenization_spaces=True)
        tokenized = tokenizer(text, truncation=True, max_length=max_length+2,
                              padding='max_length' if padding_max_length else True)
        if not with_cls:
            tokenized['input_ids'] = [t[1:-1] for t in tokenized['input_ids']]
            tokenized['attention_mask'] = [t[1:-1] for t in tokenized['attention_mask']]
        return tokenized
    return preprocess_function


def extract_embeddings(input_ids: List[List[int]], model: PreTrainedModel,
                       batch_size: int = None, mean_embeddings: bool = False) -> List[List[float]]:
    device = model.device
    model = model.eval()
    embeddings = model.get_input_embeddings()
    if batch_size is None:
        input_ids = torch.tensor(input_ids, device=device)
        embedded_inputs = embeddings(input_ids)
        if mean_embeddings:
            embedded_inputs = embedded_inputs.mean(dim=1)
    else:
        embedded_inputs = []
        for batch in divide_to_batches(input_ids, batch_size):
            batch = torch.tensor(batch, device=device)
            batch = embeddings(batch)
            if mean_embeddings:
                batch = batch.mean(dim=1)
            embedded_inputs.append(batch)
        embedded_inputs = torch.cat(embedded_inputs, dim=0)
    embedded_inputs = embedded_inputs.detach().cpu().tolist()
    return embedded_inputs


def batch_decode(input_ids: List[List[int]], tokenizer: PreTrainedTokenizer) -> List[List[str]]:
    return [tokenizer.convert_ids_to_tokens(sentence) for sentence in input_ids]


def prepare_imdb_data_loaders(tokenizer, max_length: int = 32, batch_size: int = 32, with_cls: bool = False):
    imdb = load_dataset('imdb', data_dir='data/imdb')
    ppf = create_preprocess_function(tokenizer, max_length=max_length, padding_max_length=True, with_cls=with_cls)
    tokenized_imdb = imdb.map(ppf, batched=True)
    tokenized_imdb['train'].set_format(type='torch', columns=['input_ids', 'label'])
    tokenized_imdb['test'].set_format(type='torch', columns=['input_ids', 'label'])
    train_loader = DataLoader(tokenized_imdb['train'], batch_size=batch_size, num_workers=1, shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(tokenized_imdb['test'], batch_size=batch_size, num_workers=1)
    return train_loader, test_loader


def calculate_text_covariances(train_loader, model):
    train_dataset = train_loader.dataset
    inputs_ids = train_dataset['input_ids']
    labels = train_dataset['label'].tolist()
    covs, chols = [], []
    for label in set(labels):
        lab_ids = [ind.tolist() for i, ind in enumerate(inputs_ids) if labels[i] == label]
        embeddings = torch.tensor(extract_embeddings(lab_ids, model, batch_size=128, mean_embeddings=True))
        covs.append(calculate_covariance(embeddings))
        chols.append(cholesky(covs[-1]))
    return torch.stack(covs, dim=0), torch.stack(chols, dim=0)
