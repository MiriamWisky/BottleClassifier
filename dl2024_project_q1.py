import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

base_dir = '/home/viskymi/project/DNN_course/project/data'

categories = ['Beer Bottles', 'Plastic Bottles', 'Soda Bottles', 'Water Bottles', 'Wine Bottles']

# This function defines DataLoaders for the training and validation sets, performs augmentation, normalization, and resizing of the images.
def prepare_data_loaders(base_dir, categories, batch_size=32, train_split=0.85, val_split=0.15, img_size=(224, 224), augment=True):

    if augment:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    dataset = datasets.ImageFolder(root=base_dir, transform=transform)

    train_size = int(train_split * len(dataset))
    val_size = int(val_split * len(dataset))

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    return train_loader, val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This function defines a generic CNN model with configurable layers, including convolutional layers, fully connected layers, and activation functions, allowing flexibility for different experiments.
class GenericCNN(nn.Module):
    def __init__(self, num_classes=5, conv_channels=[32, 64, 128, 256], fc_units=512, activation=nn.ReLU, dropout_prob=0.5, img_size=(224, 224)):
        super(GenericCNN, self).__init__()

        self.conv_layers = nn.ModuleList()
        in_channels = 3  

        for out_channels in conv_channels:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    activation(),  
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            )
            in_channels = out_channels  

        self.flattened_size = self._get_conv_output((3, *img_size))

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, fc_units),  
            activation(),  
            nn.Dropout(p=dropout_prob),  
            nn.Linear(fc_units, num_classes)  
        )

    def _get_conv_output(self, shape):
        o = torch.zeros(1, *shape)
        o = o.to(next(self.parameters()).device)  
        for conv_layer in self.conv_layers:
            o = conv_layer(o)
        return int(o.numel())

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)  
        x = x.view(x.size(0), -1) 
        x = self.fc_layers(x)  
        return x

model = GenericCNN(num_classes=len(categories)).to(device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This function calculates the initial loss of the model.
def check_initial_loss(model, loader, criterion):
    model.eval()
    model.to(device)  
    with torch.no_grad():
        total_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)  
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
    initial_loss = total_loss / len(loader.dataset)
    print(f'Initial Loss: {initial_loss:.4f}')

# This function calculates the initial loss of the model with regularization applied.
def check_initial_loss_with_regularization(model, loader, criterion, weight_decay=1e-4):
    model.eval()
    model.to(device) 
    total_loss = 0.0
    reg_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)  
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param) ** 2
            reg_loss = weight_decay * l2_reg
            
            total_loss += (loss.item() + reg_loss.item()) * inputs.size(0)
    
    initial_loss_with_reg = total_loss / len(loader.dataset)
    print(f'Initial Loss with Regularization: {initial_loss_with_reg:.4f}')

# This function tests the model's ability to overfit on a small number of images to verify overfitting.
def overfit_on_few_images(model, loader, optimizer, criterion, num_epochs=10):
    model.train()
    model.to(device)  
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)  
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(loader.dataset)
        accuracy = correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')

# This function trains the model on the training dataset and evaluates its performance on the validation dataset, logging the training and validation loss and accuracy after each epoch.
def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, num_epochs=30, output_dir='output', early_stopping_patience=12):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    misclassified_images = []  

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        misclassified_images = []

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0) 

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset) 
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)  
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)  

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                misclassified_indices = (predicted != labels).nonzero(as_tuple=True)[0]
                for idx in misclassified_indices:
                    global_idx = batch_idx * val_loader.batch_size + idx.item()
                    if global_idx < len(val_loader.dataset.indices):
                        original_idx = val_loader.dataset.indices[global_idx]
                        img_path, actual_label = val_loader.dataset.dataset.samples[original_idx]
                        predicted_label = predicted[idx].item()
                        misclassified_images.append((os.path.abspath(img_path), actual_label, predicted_label))
                    else:
                        print(f"Warning: global_idx {global_idx} is out of range for the dataset indices.")

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    save_results(train_losses, val_losses, train_accuracies, val_accuracies, all_labels, all_predictions, misclassified_images, output_dir)


# This function saves the model's training and validation results, including loss, accuracy, and misclassified images, to specified output files for later analysis.
def save_results(train_losses, val_losses, train_accuracies, val_accuracies, labels, predictions, misclassified_images, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{output_dir}/loss_plot.png')
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{output_dir}/accuracy_plot.png')
    plt.close()

    class_names = ['Beer Bottles', 'Plastic Bottles', 'Soda Bottles', 'Water Bottles', 'Wine Bottles']
    cm = confusion_matrix(labels, predictions)

    fig, ax = plt.subplots(figsize=(10, 7))  
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap=plt.cm.viridis, xticks_rotation=45)  
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()

    with open(f'{output_dir}/misclassified_images.txt', 'w') as f:
        for item in misclassified_images:  
            f.write(f"{item}\n")


learning_rate = 0.001
batch_size = 64
num_epochs = 30
output_dir = 'output'

train_loader, val_loader = prepare_data_loaders(base_dir, categories, batch_size)

model = GenericCNN(num_classes=len(categories))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()

check_initial_loss(model, train_loader, criterion)
check_initial_loss_with_regularization(model, train_loader, criterion)
small_dataset, _ = random_split(train_loader.dataset, [2, len(train_loader.dataset) - 2])
small_loader = DataLoader(small_dataset, batch_size=batch_size, shuffle=True)
overfit_on_few_images(model, small_loader, optimizer, criterion)


train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, num_epochs, output_dir)

# This function trains the model using different learning rates.
def train_with_different_lr(model, train_loader, validation_loader, learning_rate, experiment_index):
    print(f'\nTraining with Learning Rate: {learning_rate}')
    return train_and_evaluate(model, train_loader, validation_loader, optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4), criterion=nn.CrossEntropyLoss(), num_epochs=30, output_dir=f'output/lr1_{learning_rate}_{experiment_index}')

learning_rates = [0.005, 0.0005]
for idx, lr in enumerate(learning_rates):
    model = GenericCNN(num_classes=len(categories))
    train_with_different_lr(model, train_loader, val_loader, learning_rate=lr, experiment_index=f'lr_{idx}')

# This function trains the model using different batch sizes.
def train_with_different_batch_size(model, batch_size, experiment_index):
    print(f'\nTraining with Batch Size: {batch_size}')
    train_loader, val_loader = prepare_data_loaders(base_dir, categories, batch_size=batch_size)
    return train_and_evaluate(model, train_loader, val_loader, optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4), criterion=nn.CrossEntropyLoss(), num_epochs=30, output_dir=f'output/batch1_{batch_size}_{experiment_index}')

batch_sizes = [32, 128]
for idx, batch_size in enumerate(batch_sizes):
    model = GenericCNN(num_classes=len(categories))
    train_with_different_batch_size(model, batch_size, experiment_index=f'batch_{idx}')


# This function trains the model using different optimizers.
def train_with_different_optimizer(model, train_loader, validation_loader, optimizer_class, experiment_index):
    print(f'\nTraining with Optimizer: {optimizer_class.__name__}')
    return train_and_evaluate(model, train_loader, validation_loader, optimizer=optimizer_class(model.parameters(), lr=0.001, weight_decay=1e-4), criterion=nn.CrossEntropyLoss(), num_epochs=30, output_dir=f'output/opt1_{optimizer_class.__name__}_{experiment_index}')

optimizers = [torch.optim.Adam, torch.optim.RMSprop]
for idx, optimizer_class in enumerate(optimizers):
    model = GenericCNN(num_classes=len(categories))
    train_with_different_optimizer(model, train_loader, val_loader, optimizer_class, experiment_index=f'opt_{idx}')

# This function trains the model with and without augmentation.
def train_with_augmentation(model, use_augmentation, experiment_index):
    print(f'\nTraining with Augmentation: {use_augmentation}')
    train_loader, val_loader = prepare_data_loaders(base_dir, categories, batch_size=32, augment=use_augmentation)
    return train_and_evaluate(model, train_loader, val_loader, optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4), criterion=nn.CrossEntropyLoss(), num_epochs=30, output_dir=f'output/aug1_{use_augmentation}_{experiment_index}')

augmentations = [True, False]
for idx, use_augmentation in enumerate(augmentations):
    model = GenericCNN(num_classes=len(categories))
    train_with_augmentation(model, use_augmentation, experiment_index=f'aug_{idx}')

# This function trains the model using different configurations of convolutional layers.
def train_with_different_conv_layers(conv_channels, experiment_index):
    print(f'\nTraining with Conv Layers: {conv_channels}')
    model = GenericCNN(num_classes=len(categories), conv_channels=conv_channels)
    return train_and_evaluate(model, train_loader, val_loader, optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4), criterion=nn.CrossEntropyLoss(), num_epochs=30, output_dir=f'output/conv1_{conv_channels}_{experiment_index}')

conv_configs = [[32, 64, 128], [32, 64, 128, 256, 512]]
for idx, conv_channels in enumerate(conv_configs):
    train_with_different_conv_layers(conv_channels, experiment_index=f'conv_{idx}')


# This function trains the model using different activations.
def train_with_different_activation(activation, experiment_index):
    print(f'\nTraining with Activation: {activation.__name__}')
    model = GenericCNN(num_classes=len(categories), activation=activation)
    return train_and_evaluate(model, train_loader, val_loader, optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4), criterion=nn.CrossEntropyLoss(), num_epochs=30, output_dir=f'output/act1_{activation.__name__}_{experiment_index}')

activations = [nn.ReLU, nn.LeakyReLU]
for idx, activation in enumerate(activations):
    train_with_different_activation(activation, experiment_index=f'act_{idx}')


# This function trains the model using different image sizes.
def train_with_different_img_size(img_size, experiment_index):
    print(f'\nTraining with Image Size: {img_size}')
    train_loader, val_loader = prepare_data_loaders(base_dir, categories, img_size=img_size, batch_size=32)
    model = GenericCNN(num_classes=len(categories), img_size=img_size)
    return train_and_evaluate(model, train_loader, val_loader, optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4), criterion=nn.CrossEntropyLoss(), num_epochs=30, output_dir=f'output/img_size1_{img_size}_{experiment_index}')

img_sizes = [(128, 128), (256, 256)]
for idx, img_size in enumerate(img_sizes):
    train_with_different_img_size(img_size, experiment_index=f'img_size_{idx}')

# This function trains the model using different amount of epochs.
def train_with_different_epochs(num_epochs, experiment_index):
    print(f'\nTraining with Number of Epochs: {num_epochs}')
    model = GenericCNN(num_classes=len(categories))
    return train_and_evaluate(model, train_loader, val_loader, optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4), criterion=nn.CrossEntropyLoss(), num_epochs=num_epochs, output_dir=f'output/epochs1_{num_epochs}_{experiment_index}')

epoch_counts = [20, 50]
for idx, num_epochs in enumerate(epoch_counts):
    train_with_different_epochs(num_epochs, experiment_index=f'epochs_{idx}')

# This function trains the model using different fc units.
def train_with_different_fc_units(fc_units, experiment_index):
    print(f'\nTraining with FC Units: {fc_units}')
    model = GenericCNN(num_classes=len(categories), fc_units=fc_units)
    return train_and_evaluate(model, train_loader, val_loader, optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4), criterion=nn.CrossEntropyLoss(), num_epochs=30, output_dir=f'output/fc1_{fc_units}_{experiment_index}')

fc_units_list = [256, 819]
for idx, fc_units in enumerate(fc_units_list):
    train_with_different_fc_units(fc_units, experiment_index=f'fc_{idx}')



