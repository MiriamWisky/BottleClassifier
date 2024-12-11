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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models

base_dir = '/home/viskymi/project/DNN_course/project/data'

categories = ['Beer Bottles', 'Plastic Bottles', 'Soda Bottles', 'Water Bottles', 'Wine Bottles']

# This function defines DataLoaders for the training and validation sets, performs augmentation, normalization, and resizing of the images.
def prepare_data_loaders(base_dir, categories, batch_size=128, train_split=0.85, val_split=0.15, img_size=(224, 224), augment=True):

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

    # print("Classes:", dataset.classes)
    # print("Class to Index:", dataset.class_to_idx)

    return train_loader, val_loader



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# This class defines a model based on the ResNet50 architecture, using transfer learning with fine-tuning. The earlier layers of ResNet50 are frozen, and only the later layers are trained on the target dataset, allowing adaptation to the specific classification task.
class PretrainedResNet50_FT(nn.Module):
    def __init__(self, num_classes=5, fc_units=650, dropout_prob=0.3, activation=nn.ReLU, img_size=(224, 224)):
        super(PretrainedResNet50_FT, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

        for param in self.resnet.parameters():
            param.requires_grad = False 

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, fc_units),
            activation(), 
            nn.Dropout(dropout_prob),
            nn.Linear(fc_units, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

model = PretrainedResNet50_FT(num_classes=len(categories)).to(device)


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
def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, num_epochs=25, output_dir='output', early_stopping_patience=9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

  
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

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

        scheduler.step(val_loss)

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

learning_rate = 0.0001
batch_size = 128
num_epochs = 25
output_dir = 'output'

train_loader, val_loader = prepare_data_loaders(base_dir, categories, batch_size)
model = PretrainedResNet50_FT(num_classes=len(categories)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
check_initial_loss(model, train_loader, criterion)
check_initial_loss_with_regularization(model, train_loader, criterion)
small_dataset, _ = random_split(train_loader.dataset, [2, len(train_loader.dataset) - 2])
small_loader = DataLoader(small_dataset, batch_size=batch_size, shuffle=True)
overfit_on_few_images(model, small_loader, optimizer, criterion)


# This function creates and returns an Adam optimizer for fine-tuning a ResNet model, applying different learning rates to the final layer (fc) and the last ResNet block (layer4), with an optional weight decay for regularization.
def get_optimizer(model, base_lr, ft_lr_ratio=0.1, weight_decay=1e-5):
    optimizer = torch.optim.Adam([
        {'params': model.resnet.layer4.parameters(), 'lr': base_lr * ft_lr_ratio, 'weight_decay': weight_decay},  
        {'params': model.resnet.fc.parameters(), 'lr': base_lr, 'weight_decay': weight_decay}  
    ])
    return optimizer


# This function trains the model using different learning rates.
def train_with_different_lr(model, train_loader, validation_loader, learning_rate, experiment_index):
    print(f'\nTraining with Learning Rate: {learning_rate}')
    optimizer = get_optimizer(model, base_lr=learning_rate, ft_lr_ratio=0.1)  
    train_and_evaluate(model, train_loader, validation_loader, optimizer=optimizer, criterion=nn.CrossEntropyLoss(), num_epochs=25, output_dir=f'output/lr3_FT_{learning_rate}_{experiment_index}')

learning_rates = [0.0001, 0.001]
for idx, lr in enumerate(learning_rates):
    model = PretrainedResNet50_FT(num_classes=len(categories)).to(device)
    train_with_different_lr(model, train_loader, val_loader, learning_rate=lr, experiment_index=f'lr_{idx}')


# This function trains the model using different batch sizes.
def train_with_different_batch_size(model, batch_size, experiment_index):
    print(f'\nTraining with Batch Size: {batch_size}')
    train_loader, val_loader = prepare_data_loaders(base_dir, categories, batch_size=batch_size)
    optimizer = get_optimizer(model, base_lr=0.0001, ft_lr_ratio=0.1)  
    train_and_evaluate(model, train_loader, val_loader, optimizer=optimizer, criterion=nn.CrossEntropyLoss(), num_epochs=25, output_dir=f'output/batch3_FT_{batch_size}_{experiment_index}')

batch_sizes = [64, 32]
for idx, batch_size in enumerate(batch_sizes):
    model = PretrainedResNet50_FT(num_classes=len(categories)).to(device)
    train_with_different_batch_size(model, batch_size, experiment_index=f'batch_{idx}')


# This function trains the model using different optimizers.
def train_with_different_optimizer(model, train_loader, validation_loader, optimizer_class, experiment_index):
    print(f'\nTraining with Optimizer: {optimizer_class.__name__}')
    optimizer = optimizer_class([
        {'params': model.resnet.layer4.parameters(), 'lr': 0.0001 * 0.1, 'weight_decay': 1e-4},
        {'params': model.resnet.fc.parameters(), 'lr': 0.0001, 'weight_decay': 1e-4}
    ])  
    train_and_evaluate(model, train_loader, validation_loader, optimizer=optimizer, criterion=nn.CrossEntropyLoss(), num_epochs=25, output_dir=f'output/opt3_FT_{optimizer_class.__name__}_{experiment_index}')

optimizers = [torch.optim.Adam, torch.optim.RMSprop]
for idx, optimizer_class in enumerate(optimizers):
    model = PretrainedResNet50_FT(num_classes=len(categories)).to(device)
    train_with_different_optimizer(model, train_loader, val_loader, optimizer_class, experiment_index=f'opt_{idx}')


# This function trains the model with and without augmentation.
def train_with_augmentation(model, use_augmentation, experiment_index):
    print(f'\nTraining with Augmentation: {use_augmentation}')
    train_loader, val_loader = prepare_data_loaders(base_dir, categories, batch_size=128, augment=use_augmentation)
    optimizer = get_optimizer(model, base_lr=0.0001, ft_lr_ratio=0.1)  
    train_and_evaluate(model, train_loader, val_loader, optimizer=optimizer, criterion=nn.CrossEntropyLoss(), num_epochs=25, output_dir=f'output/aug3_FT_{use_augmentation}_{experiment_index}')

augmentations = [True, False]
for idx, use_augmentation in enumerate(augmentations):
    model = PretrainedResNet50_FT(num_classes=len(categories)).to(device)
    train_with_augmentation(model, use_augmentation, experiment_index=f'aug_{idx}')


# This function trains the model using different activations.
def train_with_different_activation(activation, experiment_index):
    print(f'\nTraining with Activation: {activation.__name__}')
    model = PretrainedResNet50_FT(num_classes=len(categories), activation=activation).to(device)
    optimizer = get_optimizer(model, base_lr=0.0001, ft_lr_ratio=0.1)  
    train_and_evaluate(model, train_loader, val_loader, optimizer=optimizer, criterion=nn.CrossEntropyLoss(), num_epochs=25, output_dir=f'output/act3_FT_{activation.__name__}_{experiment_index}')

activations = [nn.ReLU, nn.LeakyReLU]
for idx, activation in enumerate(activations):
    train_with_different_activation(activation, experiment_index=f'act_{idx}')


# This function trains the model using different image sizes.
def train_with_different_img_size(img_size, experiment_index):
    print(f'\nTraining with Image Size: {img_size}')
    train_loader, val_loader = prepare_data_loaders(base_dir, categories, img_size=img_size, batch_size=128)
    model = PretrainedResNet50_FT(num_classes=len(categories), img_size=img_size).to(device)
    optimizer = get_optimizer(model, base_lr=0.0001, ft_lr_ratio=0.1)  
    train_and_evaluate(model, train_loader, val_loader, optimizer=optimizer, criterion=nn.CrossEntropyLoss(), num_epochs=25, output_dir=f'output/img_size3_FT_{img_size}_{experiment_index}')

img_sizes = [(128, 128), (256, 256)]
for idx, img_size in enumerate(img_sizes):
    train_with_different_img_size(img_size, experiment_index=f'img_size_{idx}')


# This function trains the model using different amount of epochs.
def train_with_different_epochs(num_epochs, experiment_index):
    print(f'\nTraining with Number of Epochs: {num_epochs}')
    model = PretrainedResNet50_FT(num_classes=len(categories)).to(device)
    optimizer = get_optimizer(model, base_lr=0.0001, ft_lr_ratio=0.1)  
    train_and_evaluate(model, train_loader, val_loader, optimizer=optimizer, criterion=nn.CrossEntropyLoss(), num_epochs=num_epochs, output_dir=f'output/epochs3_FT_{num_epochs}_{experiment_index}')

epoch_counts = [20, 50]
for idx, num_epochs in enumerate(epoch_counts):
    train_with_different_epochs(num_epochs, experiment_index=f'epochs_{idx}')


# This function trains the model using different fc units.
def train_with_different_fc_units(fc_units, experiment_index):
    print(f'\nTraining with FC Units: {fc_units}')
    model = PretrainedResNet50_FT(num_classes=len(categories), fc_units=fc_units).to(device)
    optimizer = get_optimizer(model, base_lr=0.0001, ft_lr_ratio=0.1)  
    train_and_evaluate(model, train_loader, val_loader, optimizer=optimizer, criterion=nn.CrossEntropyLoss(), num_epochs=25, output_dir=f'output/fc3_FT_{fc_units}_{experiment_index}')

fc_units_list = [512, 819]
for idx, fc_units in enumerate(fc_units_list):
    train_with_different_fc_units(fc_units, experiment_index=f'fc_{idx}')
