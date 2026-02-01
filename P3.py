import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import itertools
import random
import time
import numpy as np

# Set random seeds for reproducibility
def set_seed(seed=40):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(40)

# Set device to MPS if available.
device = torch.device("mps") # IMPORTANT: If running on a Mac with Apple Silicon, use "mps", if using nvidia GPU, change to "cuda"
print("Using device:", device)
print()

# Tuning hyperparameters
hyperparameters = {
    'learning_rate': [0.01, 0.001, 0.0001],
    'batch_size': [32, 64, 128],
    'optimizer': ['SGD', 'Adam'],
    'dropout_rate': [0.2, 0.5],
}

# Function to create data loaders, with appropriate batch sizes
def create_data_loaders(train_data, val_data, batch_size):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# For each model architecture, tune hyperparameters using the validation set.
def training_and_validation(model, train_loader, val_loader, learning_rate, optimizer_name):
    
    # Early stopping parameters
    patience = 4  # Number of epochs to wait for improvement
    best_val_loss = float('inf')  
    no_improvement_counter = 0  

    model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop with early stopping
    for epoch in range(25):  
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total

        #print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
            if no_improvement_counter >= patience:
                break

    return accuracy, epoch + 1 

# retrain the final model on the combined training and validation data, and report test accuracy on the designated test set.
def final_training_and_testing(model, train_loader, test_loader, learning_rate, optimizer_name):
    
    model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(25):  
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Testing loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy, epoch + 1  # epoch should be 25 here since we train for full 25 epochs


# Generate all possible combinations
all_combinations = list(itertools.product(
    hyperparameters['learning_rate'],
    hyperparameters['batch_size'],
    hyperparameters['optimizer'],
    hyperparameters['dropout_rate']
))
# Randomly sample 12 combinations
sampled_combinations = random.sample(all_combinations, 12)


####### MLP for MNIST ######


# Evaluate at least three distinct architectures: 1. Shallow (1 hidden layer, e.g., 128 units), 2. Medium-depth (3 hidden layers, e.g., [512, 256, 128]), 3. Deep (at least 5 hidden layers, your choice)


# Use PyTorch (torchvision) to load MNIST datasets.
# Normalize pixel values to [0,1] is handled by transforms.ToTensor()
# Define transformations; here we just convert images to tensors; This also normalizes pixel values to [0,1]
mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])

# we downloaded the training sets for MNIST datasets, and applied the transformations
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)

# Validation data split 50,000 training samples and 10,000 validation samples for MNIST
mnist_train_dataset, mnist_validation_dataset = torch.utils.data.random_split(mnist_dataset, [50000, 10000], generator=torch.Generator().manual_seed(40))


# For all architectures, tune hyperparameters using the validation set. 
best_accuracy_shallow_mlp = 0
best_hyperparams_shallow_mlp = {}
best_accuracy_medium_mlp = 0
best_hyperparams_medium_mlp = {}
best_accuracy_deep_mlp = 0
best_hyperparams_deep_mlp = {}

print("For MLP evaluations on MNIST:")
print()

for learning_rate, batch_size, optimizer_name, dropout_rate in sampled_combinations:

    mnist_train_loader, mnist_val_loader = create_data_loaders(mnist_train_dataset, mnist_validation_dataset, batch_size)

    # Shallow MLP
    shallow_mlp = torch.nn.Sequential(
        torch.nn.Linear(784, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(128, 10)
    )

    shallow_mlp.to(device)

    time_start = time.time()
    accuracy_shallow_mlp, epoch = training_and_validation(
        model=shallow_mlp,
        train_loader=mnist_train_loader,
        val_loader=mnist_val_loader,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name
    )
    time_end = time.time()
    print(f"MLP (shallow, 1 hidden): Learning Rate: {learning_rate}, Batch Size: {batch_size}, Optimizer: {optimizer_name}, Dropout: {dropout_rate}, Accuracy: {accuracy_shallow_mlp:.2f}%, Time: {(time_end - time_start) / 60:.2f} minutes, Epoch: {epoch}")

    if accuracy_shallow_mlp > best_accuracy_shallow_mlp:
        best_accuracy_shallow_mlp = accuracy_shallow_mlp
        best_hyperparams_shallow_mlp = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizer': optimizer_name,
            'dropout_rate': dropout_rate
        }


    # Medium-depth MLP
    medium_mlp = torch.nn.Sequential(
        torch.nn.Linear(784, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(128, 10)
    )

    medium_mlp.to(device)

    time_start = time.time()
    accuracy_medium_mlp, epoch = training_and_validation(
        model=medium_mlp,
        train_loader=mnist_train_loader,
        val_loader=mnist_val_loader,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name
    )
    time_end = time.time()
    print(f"MLP (medium, 3 hidden): Learning Rate: {learning_rate}, Batch Size: {batch_size}, Optimizer: {optimizer_name}, Dropout: {dropout_rate}, Accuracy: {accuracy_medium_mlp:.2f}%, Time: {(time_end - time_start) / 60:.2f} minutes, Epoch: {epoch}")

    if accuracy_medium_mlp > best_accuracy_medium_mlp:
        best_accuracy_medium_mlp = accuracy_medium_mlp
        best_hyperparams_medium_mlp = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizer': optimizer_name,
            'dropout_rate': dropout_rate
        }

    # Deep MLP (5 hidden layers)
    deep_mlp = torch.nn.Sequential(
        torch.nn.Linear(784, 1024),  
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(1024, 512),  
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(64, 10)
    )
    deep_mlp.to(device)

    time_start = time.time()
    accuracy_deep_mlp, epoch = training_and_validation(
        model=deep_mlp,
        train_loader=mnist_train_loader,
        val_loader=mnist_val_loader,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name
    )
    time_end = time.time()
    print(f"MLP (deep, >=5 hidden): Learning Rate: {learning_rate}, Batch Size: {batch_size}, Optimizer: {optimizer_name}, Dropout: {dropout_rate}, Accuracy: {accuracy_deep_mlp:.2f}%, Time: {(time_end - time_start) / 60:.2f} minutes, Epoch: {epoch}")

    if accuracy_deep_mlp > best_accuracy_deep_mlp:
        best_accuracy_deep_mlp = accuracy_deep_mlp
        best_hyperparams_deep_mlp = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizer': optimizer_name,
            'dropout_rate': dropout_rate
        }
    print()

print(f"Completed MLP evaluations on MNIST.")
print(f"Best Shallow MLP Hyperparameters - Batch Size: {best_hyperparams_shallow_mlp['batch_size']}, Learning Rate: {best_hyperparams_shallow_mlp['learning_rate']}, Optimizer: {best_hyperparams_shallow_mlp['optimizer']}, Dropout: {best_hyperparams_shallow_mlp['dropout_rate']}")
print(f"Best Medium MLP Hyperparameters - Batch Size: {best_hyperparams_medium_mlp['batch_size']}, Learning Rate: {best_hyperparams_medium_mlp['learning_rate']}, Optimizer: {best_hyperparams_medium_mlp['optimizer']}, Dropout: {best_hyperparams_medium_mlp['dropout_rate']}")
print(f"Best Deep MLP Hyperparameters - Batch Size: {best_hyperparams_deep_mlp['batch_size']}, Learning Rate: {best_hyperparams_deep_mlp['learning_rate']}, Optimizer: {best_hyperparams_deep_mlp['optimizer']}, Dropout: {best_hyperparams_deep_mlp['dropout_rate']}")
print()

#Combine train and validation sets for final training on MNIST with best hyperparameters found for each architecture
final_mnist_train_dataset = torch.utils.data.ConcatDataset([mnist_train_dataset, mnist_validation_dataset])

# Getting the Test set for MNIST
mnist_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)

# Do the final evaluation on the test set for MNIST using the best hyperparameters for each architecture.

#### For Shallow MLP
final_mnist_train_loader_shallow_mlp = torch.utils.data.DataLoader(
    final_mnist_train_dataset,
    batch_size=best_hyperparams_shallow_mlp['batch_size'],  # Using best batch size from shallow MLP
    shuffle=True,
)

mnist_test_loader_shallow_mlp = torch.utils.data.DataLoader(
    mnist_test_dataset,
    batch_size=best_hyperparams_shallow_mlp['batch_size'],  # Using best batch size from shallow MLP 
    shuffle=False,
)

# Shallow MLP
shallow_mlp = torch.nn.Sequential(
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(best_hyperparams_shallow_mlp['dropout_rate']),
    torch.nn.Linear(128, 10)
)

shallow_mlp.to(device)

time_start = time.time()
final_accuracy_shallow_mlp, epoch = final_training_and_testing(
        model=shallow_mlp,
        train_loader=final_mnist_train_loader_shallow_mlp,
        test_loader=mnist_test_loader_shallow_mlp,
        learning_rate=best_hyperparams_shallow_mlp['learning_rate'],
        optimizer_name=best_hyperparams_shallow_mlp['optimizer']
    )
time_end = time.time()

print(f"Final Test Accuracy for Shallow MLP: {final_accuracy_shallow_mlp:.2f}%")
print(f"Runtime for Shallow MLP on MNIST dataset in minutes: {(time_end - time_start) / 60:.2f}")
print(f"Total Epochs for Shallow MLP on MNIST: {epoch}")
print()

#### For Medium MLP
final_mnist_train_loader_medium_mlp = torch.utils.data.DataLoader(
    final_mnist_train_dataset,
    batch_size=best_hyperparams_medium_mlp['batch_size'],  # Using best batch size from medium MLP
    shuffle=True,
)
mnist_test_loader_medium_mlp = torch.utils.data.DataLoader(
    mnist_test_dataset,
    batch_size=best_hyperparams_medium_mlp['batch_size'],  # Using best batch size from medium MLP as an example
    shuffle=False,
)

# Medium-depth MLP
medium_mlp = torch.nn.Sequential(
    torch.nn.Linear(784, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(best_hyperparams_medium_mlp['dropout_rate']),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(best_hyperparams_medium_mlp['dropout_rate']),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(best_hyperparams_medium_mlp['dropout_rate']),
    torch.nn.Linear(128, 10)
)
medium_mlp.to(device)

time_start = time.time()
final_accuracy_medium_mlp, epoch = final_training_and_testing(
        model=medium_mlp,
        train_loader=final_mnist_train_loader_medium_mlp,
        test_loader=mnist_test_loader_medium_mlp,
        learning_rate=best_hyperparams_medium_mlp['learning_rate'],
        optimizer_name=best_hyperparams_medium_mlp['optimizer']
)
time_end = time.time()

print(f"Final Test Accuracy for Medium MLP: {final_accuracy_medium_mlp:.2f}%")
print(f"Runtime for Medium MLP on MNIST dataset in minutes: {(time_end - time_start) / 60:.2f}")
print(f"Total Epochs for Medium MLP on MNIST: {epoch}")
print()

# For Deep MLP
mnist_test_loader_deep_mlp = torch.utils.data.DataLoader(
    mnist_test_dataset,
    batch_size=best_hyperparams_deep_mlp['batch_size'],  # Using best batch size from deep MLP as an example
    shuffle=False,
)

final_mnist_train_loader_deep_mlp = torch.utils.data.DataLoader(
    final_mnist_train_dataset,
    batch_size=best_hyperparams_deep_mlp['batch_size'],  # Using best batch size from deep MLP
    shuffle=True,
)

# Deep MLP
deep_mlp = torch.nn.Sequential(
    torch.nn.Linear(784, 1024),  
    torch.nn.ReLU(),
    torch.nn.Dropout(best_hyperparams_deep_mlp['dropout_rate']),
    torch.nn.Linear(1024, 512),  
    torch.nn.ReLU(),
    torch.nn.Dropout(best_hyperparams_deep_mlp['dropout_rate']),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(best_hyperparams_deep_mlp['dropout_rate']),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(best_hyperparams_deep_mlp['dropout_rate']),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Dropout(best_hyperparams_deep_mlp['dropout_rate']),
    torch.nn.Linear(64, 10)
)

deep_mlp.to(device)

time_start = time.time()
final_accuracy_deep_mlp, epoch = final_training_and_testing(
        model=deep_mlp,
        train_loader=final_mnist_train_loader_deep_mlp,
        test_loader=mnist_test_loader_deep_mlp,
        learning_rate=best_hyperparams_deep_mlp['learning_rate'],
        optimizer_name=best_hyperparams_deep_mlp['optimizer']
)
time_end = time.time()
print(f"Final Test Accuracy for Deep MLP: {final_accuracy_deep_mlp:.2f}%")
print(f"Runtime for Deep MLP on MNIST dataset in minutes: {(time_end - time_start) / 60:.2f}")
print(f"Total Epochs for Deep MLP on MNIST: {epoch}")
print()



######  Multilayer Perceptrons (MLPs) for CIFAR-10  ######

# Define transformations; here we just convert images to tensors; This also normalizes pixel values to [0,1]
cifar10_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])

# we downloaded the training sets for CIFAR-10 datasets, and applied the transformations
cifar10_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar10_transform)

# Getting the Test set for CIFAR-10
cifar10_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar10_transform)

# Validation data split 45,000 training samples and 5,000 validation samples for CIFAR-10
cifar10_train_dataset, cifar10_validation_dataset = torch.utils.data.random_split(cifar10_dataset, [45000, 5000], generator=torch.Generator().manual_seed(40))

# For all architectures, tune hyperparameters using the validation set. 
best_accuracy_shallow_mlp = 0
best_hyperparams_shallow_mlp = {}
best_accuracy_medium_mlp = 0
best_hyperparams_medium_mlp = {}
best_accuracy_deep_mlp = 0
best_hyperparams_deep_mlp = {}

print("For MLP evaluations on CIFAR-10:")
print()

for learning_rate, batch_size, optimizer_name, dropout_rate in sampled_combinations:

    cifar10_train_loader, cifar10_val_loader = create_data_loaders(cifar10_train_dataset, cifar10_validation_dataset, batch_size)

    # Shallow MLP
    shallow_mlp = torch.nn.Sequential(
        torch.nn.Linear(3072, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(128, 10)
    )

    shallow_mlp.to(device)

    time_start = time.time()
    accuracy_shallow_mlp, epoch = training_and_validation(
        model=shallow_mlp,
        train_loader=cifar10_train_loader,
        val_loader=cifar10_val_loader,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name
    )
    time_end = time.time()
    print(f"MLP (shallow, 1 hidden): Learning Rate: {learning_rate}, Batch Size: {batch_size}, Optimizer: {optimizer_name}, Dropout: {dropout_rate}, Accuracy: {accuracy_shallow_mlp:.2f}%, Time: {(time_end - time_start) / 60:.2f} minutes, Epoch: {epoch}")

    if accuracy_shallow_mlp > best_accuracy_shallow_mlp:
        best_accuracy_shallow_mlp = accuracy_shallow_mlp
        best_hyperparams_shallow_mlp = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizer': optimizer_name,
            'dropout_rate': dropout_rate
        }

    # Medium-depth MLP
    medium_mlp = torch.nn.Sequential(
        torch.nn.Linear(3072, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(128, 10)
    )

    medium_mlp.to(device)

    time_start = time.time()
    accuracy_medium_mlp, epoch = training_and_validation(
        model=medium_mlp,
        train_loader=cifar10_train_loader,
        val_loader=cifar10_val_loader,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name
    )
    time_end = time.time()
    print(f"MLP (medium, 3 hidden): Learning Rate: {learning_rate}, Batch Size: {batch_size}, Optimizer: {optimizer_name}, Dropout: {dropout_rate}, Accuracy: {accuracy_medium_mlp:.2f}%, Time: {(time_end - time_start) / 60:.2f} minutes, Epoch: {epoch}")

    if accuracy_medium_mlp > best_accuracy_medium_mlp:
        best_accuracy_medium_mlp = accuracy_medium_mlp
        best_hyperparams_medium_mlp = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizer': optimizer_name,
            'dropout_rate': dropout_rate
        }

    # Deep MLP (5 hidden layers)
    deep_mlp = torch.nn.Sequential(
        torch.nn.Linear(3072, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(64, 10)
    )
    
    deep_mlp.to(device)

    time_start = time.time()
    accuracy_deep_mlp, epoch = training_and_validation(
        model=deep_mlp,
        train_loader=cifar10_train_loader,
        val_loader=cifar10_val_loader,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name
    )
    time_end = time.time()
    print(f"MLP (deep, >=5 hidden): Learning Rate: {learning_rate}, Batch Size: {batch_size}, Optimizer: {optimizer_name}, Dropout: {dropout_rate}, Accuracy: {accuracy_deep_mlp:.2f}%, Time: {(time_end - time_start) / 60:.2f} minutes, Epoch: {epoch}")

    if accuracy_deep_mlp > best_accuracy_deep_mlp:
        best_accuracy_deep_mlp = accuracy_deep_mlp
        best_hyperparams_deep_mlp = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizer': optimizer_name,
            'dropout_rate': dropout_rate
        }
    print()

print(f"Completed MLP evaluations on CIFAR-10.")
print(f"Best Shallow MLP Hyperparameters - Batch Size: {best_hyperparams_shallow_mlp['batch_size']}, Learning Rate: {best_hyperparams_shallow_mlp['learning_rate']}, Optimizer: {best_hyperparams_shallow_mlp['optimizer']}, Dropout: {best_hyperparams_shallow_mlp['dropout_rate']}")
print(f"Best Medium MLP Hyperparameters - Batch Size: {best_hyperparams_medium_mlp['batch_size']}, Learning Rate: {best_hyperparams_medium_mlp['learning_rate']}, Optimizer: {best_hyperparams_medium_mlp['optimizer']}, Dropout: {best_hyperparams_medium_mlp['dropout_rate']}")
print(f"Best Deep MLP Hyperparameters - Batch Size: {best_hyperparams_deep_mlp['batch_size']}, Learning Rate: {best_hyperparams_deep_mlp['learning_rate']}, Optimizer: {best_hyperparams_deep_mlp['optimizer']}, Dropout: {best_hyperparams_deep_mlp['dropout_rate']}")
print()

#Combine train and validation sets for final training on CIFAR-10 with best hyperparameters found for each architecture
final_cifar10_train_dataset = torch.utils.data.ConcatDataset([cifar10_train_dataset, cifar10_validation_dataset])

# Do the final evaluation on the test set for CIFAR-10 using the best hyperparameters for each architecture.

##### For Shallow MLP
final_cifar10_train_loader_shallow_mlp = torch.utils.data.DataLoader(
    final_cifar10_train_dataset,
    batch_size=best_hyperparams_shallow_mlp['batch_size'],  # Using best batch size from shallow MLP
    shuffle=True,
)

cifar10_test_loader_shallow_mlp = torch.utils.data.DataLoader(
    cifar10_test_dataset,
    batch_size=best_hyperparams_shallow_mlp['batch_size'],  # Using best batch size from shallow MLP 
    shuffle=False,
)

# Shallow MLP
shallow_mlp = torch.nn.Sequential(
    torch.nn.Linear(3072, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(best_hyperparams_shallow_mlp['dropout_rate']),
    torch.nn.Linear(128, 10)
)

shallow_mlp.to(device)

time_start = time.time()
final_accuracy_shallow_mlp, epoch = final_training_and_testing(
        model=shallow_mlp,
        train_loader=final_cifar10_train_loader_shallow_mlp,
        test_loader=cifar10_test_loader_shallow_mlp,
        learning_rate=best_hyperparams_shallow_mlp['learning_rate'],
        optimizer_name=best_hyperparams_shallow_mlp['optimizer']
    )
time_end = time.time()
print(f"Final Test Accuracy for Shallow MLP on CIFAR-10: {final_accuracy_shallow_mlp:.2f}%")
print(f"Runtime for Shallow MLP on CIFAR-10 dataset in minutes: {(time_end - time_start) / 60:.2f}")
print(f"Total Epochs for Shallow MLP on CIFAR-10: {epoch}")
print()

##### For Medium MLP
final_cifar10_train_loader_medium_mlp = torch.utils.data.DataLoader(
    final_cifar10_train_dataset,
    batch_size=best_hyperparams_medium_mlp['batch_size'],  # Using best batch size from medium MLP
    shuffle=True,
)
cifar10_test_loader_medium_mlp = torch.utils.data.DataLoader(
    cifar10_test_dataset,
    batch_size=best_hyperparams_medium_mlp['batch_size'],  # Using best batch size from medium MLP as an example
    shuffle=False,
)

# Medium-depth MLP
medium_mlp = torch.nn.Sequential(
    torch.nn.Linear(3072, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(best_hyperparams_medium_mlp['dropout_rate']),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(best_hyperparams_medium_mlp['dropout_rate']),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(best_hyperparams_medium_mlp['dropout_rate']),
    torch.nn.Linear(128, 10)
 )

medium_mlp.to(device)


time_start = time.time()
final_accuracy_medium_mlp, epoch = final_training_and_testing(
        model=medium_mlp,
        train_loader=final_cifar10_train_loader_medium_mlp,
        test_loader=cifar10_test_loader_medium_mlp,
        learning_rate=best_hyperparams_medium_mlp['learning_rate'],
        optimizer_name=best_hyperparams_medium_mlp['optimizer']
)
time_end = time.time()

print(f"Final Test Accuracy for Medium MLP on CIFAR-10: {final_accuracy_medium_mlp:.2f}%")
print(f"Runtime for Medium MLP on CIFAR-10 dataset in minutes: {(time_end - time_start) / 60:.2f}")
print(f"Total Epochs for Medium MLP on CIFAR-10: {epoch}")
print()

# For Deep MLP
cifar10_test_loader_deep_mlp = torch.utils.data.DataLoader(
    cifar10_test_dataset,
    batch_size=best_hyperparams_deep_mlp['batch_size'],  # Using best batch size from deep MLP as an example
    shuffle=False,
)

final_cifar10_train_loader_deep_mlp = torch.utils.data.DataLoader(
    final_cifar10_train_dataset,
    batch_size=best_hyperparams_deep_mlp['batch_size'],  # Using best batch size from deep MLP
    shuffle=True,
)

 # Deep MLP (5 hidden layers)
deep_mlp = torch.nn.Sequential(
    torch.nn.Linear(3072, 1024),
    torch.nn.ReLU(),
    torch.nn.Dropout(best_hyperparams_deep_mlp['dropout_rate']),
    torch.nn.Linear(1024, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(best_hyperparams_deep_mlp['dropout_rate']),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(best_hyperparams_deep_mlp['dropout_rate']),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(best_hyperparams_deep_mlp['dropout_rate']),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Dropout(best_hyperparams_deep_mlp['dropout_rate']),
    torch.nn.Linear(64, 10)
)
deep_mlp.to(device)

time_start = time.time()
final_accuracy_deep_mlp, epoch = final_training_and_testing(
        model=deep_mlp,
        train_loader=final_cifar10_train_loader_deep_mlp,
        test_loader=cifar10_test_loader_deep_mlp,
        learning_rate=best_hyperparams_deep_mlp['learning_rate'],
        optimizer_name=best_hyperparams_deep_mlp['optimizer']
)
time_end = time.time()
print(f"Final Test Accuracy for Deep MLP on CIFAR-10: {final_accuracy_deep_mlp:.2f}%")
print(f"Runtime for Deep MLP on CIFAR-10 dataset in minutes: {(time_end - time_start) / 60:.2f}")
print(f"Total Epochs for Deep MLP on CIFAR-10: {epoch}")
print()


####### CNN for MNIST ######

mnist_transform = transforms.Compose([transforms.ToTensor()])

mnist_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
mnist_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)

# Validation data split 50,000 training samples and 10,000 validation samples for MNIST
mnist_train_dataset, mnist_val_dataset = torch.utils.data.random_split(mnist_train_dataset, [50000, 10000], generator=torch.Generator().manual_seed(40))

# For all architectures, tune hyperparameters using the validation set. 
best_accuracy_baseline_cnn = 0
best_hyperparams_baseline_cnn = {}
best_accuracy_enhanced_cnn = 0
best_hyperparams_enhanced_cnn = {}
best_accuracy_deep_cnn = 0
best_hyperparams_deep_cnn = {}

print("For CNN evaluations on MNIST:")
print()
for learning_rate, batch_size, optimizer_name, dropout_rate in sampled_combinations:

    cnn_train_loader, cnn_validation_loader = create_data_loaders(mnist_train_dataset, mnist_val_dataset, batch_size)

    # Baseline CNN: 2 convolutional layers + pooling, fully connected layer
    baseline_cnn = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input channels: 1 (grayscale), Output channels: 32
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),  

        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Input channels: 32, Output channels: 64
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2), 

        torch.nn.Flatten(),  # Flatten the output for the fully connected layer
        torch.nn.Linear(64 * 7 * 7, 128),  
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)  # Output layer for 10 classes
    )

    # Baseline CNN
    baseline_cnn.to(device)

    time_start = time.time()
    accuracy_baseline_cnn, epoch = training_and_validation(
        model=baseline_cnn,
        train_loader=cnn_train_loader,
        val_loader=cnn_validation_loader,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name
    )
    time_end = time.time()
    print(f"CNN (baseline, 2 conv): Learning Rate: {learning_rate}, Batch Size: {batch_size}, Optimizer: {optimizer_name}, Dropout: {dropout_rate}, Accuracy: {accuracy_baseline_cnn:.2f}%, Time: {(time_end - time_start) / 60:.2f} minutes, Epoch: {epoch}")

    if accuracy_baseline_cnn > best_accuracy_baseline_cnn:
        best_accuracy_baseline_cnn = accuracy_baseline_cnn
        best_hyperparams_baseline_cnn = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizer': optimizer_name,
            'dropout_rate': dropout_rate
        }


    # Enhanced CNN: Add batch normalization and dropout
    enhanced_cnn = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, kernel_size=3,padding=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Dropout(dropout_rate),

        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Dropout(dropout_rate),

        torch.nn.Flatten(),
        torch.nn.Linear(64 * 7 * 7, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(128, 10)
    )

    # Enhanced CNN
    enhanced_cnn.to(device)

    time_start = time.time()
    accuracy_enhanced_cnn, epoch = training_and_validation(
        model=enhanced_cnn,
        train_loader=cnn_train_loader,
        val_loader=cnn_validation_loader,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name
    )
    time_end = time.time()
    print(f"CNN (enhanced, BN+dropout): Learning Rate: {learning_rate}, Batch Size: {batch_size}, Optimizer: {optimizer_name}, Dropout: {dropout_rate}, Accuracy: {accuracy_enhanced_cnn:.2f}%, Time: {(time_end - time_start) / 60:.2f} minutes, Epoch: {epoch}")

    if accuracy_enhanced_cnn > best_accuracy_enhanced_cnn:
        best_accuracy_enhanced_cnn = accuracy_enhanced_cnn
        best_hyperparams_enhanced_cnn = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizer': optimizer_name,
            'dropout_rate': dropout_rate
        }


    # Deeper CNN: At least 3 convolutional layers with pooling, normalization, dropout, etc.
    deeper_cnn = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Dropout(dropout_rate),

        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Dropout(dropout_rate),

        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Dropout(dropout_rate),

        torch.nn.Flatten(),
        torch.nn.Linear(128 * 3 * 3, 256),  
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(256, 10)
    )

    # Deeper CNN
    deeper_cnn.to(device)

    time_start = time.time()
    accuracy_deeper_cnn, epoch = training_and_validation(
        model=deeper_cnn,
        train_loader=cnn_train_loader,
        val_loader=cnn_validation_loader,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name
    )
    time_end = time.time()
    print(f"CNN (deeper, >=3 conv): Learning Rate: {learning_rate}, Batch Size: {batch_size}, Optimizer: {optimizer_name}, Dropout: {dropout_rate}, Accuracy: {accuracy_deeper_cnn:.2f}%, Time: {(time_end - time_start) / 60:.2f} minutes, Epoch: {epoch}")

    if accuracy_deeper_cnn > best_accuracy_deep_cnn:
        best_accuracy_deep_cnn = accuracy_deeper_cnn
        best_hyperparams_deep_cnn = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizer': optimizer_name,
            'dropout_rate': dropout_rate
        }
    print()

print(f"Completed CNN evaluations on MNIST.")
print(f"Best hyperparameters for Baseline CNN - Batch Size: {best_hyperparams_baseline_cnn['batch_size']}, Learning Rate: {best_hyperparams_baseline_cnn['learning_rate']}, Optimizer: {best_hyperparams_baseline_cnn['optimizer']}, Dropout: {best_hyperparams_baseline_cnn['dropout_rate']}")
print(f"Best hyperparameters for Enhanced CNN - Batch Size: {best_hyperparams_enhanced_cnn['batch_size']}, Learning Rate: {best_hyperparams_enhanced_cnn['learning_rate']}, Optimizer: {best_hyperparams_enhanced_cnn['optimizer']}, Dropout: {best_hyperparams_enhanced_cnn['dropout_rate']}")
print(f"Best hyperparameters for Deeper CNN - Batch Size: {best_hyperparams_deep_cnn['batch_size']}, Learning Rate: {best_hyperparams_deep_cnn['learning_rate']}, Optimizer: {best_hyperparams_deep_cnn['optimizer']}, Dropout: {best_hyperparams_deep_cnn['dropout_rate']}")
print()


#Combine train and validation sets for final training
mnist_full_train_dataset_cnn = torch.utils.data.ConcatDataset([mnist_train_dataset, mnist_val_dataset])

# Do the final evaluation on the test set for MNIST using the best hyperparameters for each architecture.

##### For baseline CNN
final_mnist_train_loader_baseline_cnn = torch.utils.data.DataLoader(mnist_full_train_dataset_cnn, batch_size=best_hyperparams_baseline_cnn['batch_size'], shuffle=True)

test_loader_baseline_cnn = torch.utils.data.DataLoader(mnist_test_dataset, batch_size=best_hyperparams_baseline_cnn['batch_size'], shuffle=False)

# Baseline CNN: 2 convolutional layers + pooling, fully connected layer
baseline_cnn = torch.nn.Sequential(
    torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input channels: 1 (grayscale), Output channels: 32
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),  

    torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Input channels: 32, Output channels: 64
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),  

    torch.nn.Flatten(),  # Flatten the output for the fully connected layer
    torch.nn.Linear(64 * 7 * 7, 128),  
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)  # Output layer for 10 classes
)

baseline_cnn.to(device)

time_start = time.time()
final_accuracy_baseline_cnn, epoch = final_training_and_testing(
        model=baseline_cnn,
        train_loader=final_mnist_train_loader_baseline_cnn,
        test_loader=test_loader_baseline_cnn,
        learning_rate=best_hyperparams_baseline_cnn['learning_rate'],
        optimizer_name=best_hyperparams_baseline_cnn['optimizer']
    )
time_end = time.time()

print(f"Final Test Accuracy for Baseline CNN: {final_accuracy_baseline_cnn:.2f}%")
print(f"Runtime for Baseline CNN on MNIST dataset in minutes: {(time_end - time_start) / 60:.2f}")
print(f"Total Epochs for Baseline CNN on MNIST: {epoch}")
print()

##### For enhanced CNN
final_mnist_train_loader_enhanced_cnn = torch.utils.data.DataLoader(mnist_full_train_dataset_cnn, batch_size=best_hyperparams_enhanced_cnn['batch_size'], shuffle=True)

# For testing
test_loader_enhanced_cnn = torch.utils.data.DataLoader(mnist_test_dataset, batch_size=best_hyperparams_enhanced_cnn['batch_size'], shuffle=False)
# Enhanced CNN: Add batch normalization and dropout
enhanced_cnn = torch.nn.Sequential(
    torch.nn.Conv2d(1, 32, kernel_size=3,padding=1),
    torch.nn.BatchNorm2d(32),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Dropout(best_hyperparams_enhanced_cnn['dropout_rate']),

    torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(64),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Dropout(best_hyperparams_enhanced_cnn['dropout_rate']),

    torch.nn.Flatten(),
    torch.nn.Linear(64 * 7 * 7, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(best_hyperparams_enhanced_cnn['dropout_rate']),
    torch.nn.Linear(128, 10)
)

enhanced_cnn.to(device)

time_start = time.time()
final_accuracy_enhanced_cnn, epoch = final_training_and_testing(
        model=enhanced_cnn,
        train_loader=final_mnist_train_loader_enhanced_cnn,
        test_loader=test_loader_enhanced_cnn,
        learning_rate=best_hyperparams_enhanced_cnn['learning_rate'],
        optimizer_name=best_hyperparams_enhanced_cnn['optimizer']
    )
time_end = time.time()

print(f"Final Test Accuracy for Enhanced CNN: {final_accuracy_enhanced_cnn:.2f}%")
print(f"Runtime for Enhanced CNN on MNIST dataset in minutes: {(time_end - time_start) / 60:.2f}")
print(f"Total Epochs for Enhanced CNN on MNIST: {epoch}")
print()

##### For deeper CNN
final_mnist_train_loader_deep_cnn = torch.utils.data.DataLoader(mnist_full_train_dataset_cnn, batch_size=best_hyperparams_deep_cnn['batch_size'], shuffle=True)
# For testing
test_loader_deep_cnn = torch.utils.data.DataLoader(mnist_test_dataset, batch_size=best_hyperparams_deep_cnn['batch_size'], shuffle=False)

# Deeper CNN: At least 3 convolutional layers with pooling, normalization, dropout, etc.
deeper_cnn = torch.nn.Sequential(
    torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(32),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Dropout(best_hyperparams_deep_cnn['dropout_rate']),

    torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(64),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Dropout(best_hyperparams_deep_cnn['dropout_rate']),

    torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(128),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Dropout(best_hyperparams_deep_cnn['dropout_rate']),

    torch.nn.Flatten(),
    torch.nn.Linear(128 * 3 * 3, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(best_hyperparams_deep_cnn['dropout_rate']),
    torch.nn.Linear(256, 10)
)

deeper_cnn.to(device)

time_start = time.time()
final_accuracy_deep_cnn, epoch = final_training_and_testing(
        model=deeper_cnn,
        train_loader=final_mnist_train_loader_deep_cnn,
        test_loader=test_loader_deep_cnn,
        learning_rate=best_hyperparams_deep_cnn['learning_rate'],
        optimizer_name=best_hyperparams_deep_cnn['optimizer']
    )
time_end = time.time()
print(f"Final Test Accuracy for Deeper CNN: {final_accuracy_deep_cnn:.2f}%")
print(f"Runtime for Deeper CNN on MNIST dataset in minutes: {(time_end - time_start) / 60:.2f}")
print(f"Total Epochs for Deeper CNN on MNIST: {epoch}")
print()



####### CNN for CIFAR-10 ######

cifar_transform = transforms.Compose([transforms.ToTensor()])

cifar_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
cifar_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)

# Validation data split 45,000 training samples and 5,000 validation samples for CIFAR-10
cifar_train_dataset, cifar_val_dataset = torch.utils.data.random_split(cifar_train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(40))

# For all architectures, tune hyperparameters using the validation set. 
best_accuracy_baseline_cnn_cifar = 0
best_hyperparams_baseline_cnn_cifar = {}
best_accuracy_enhanced_cnn_cifar = 0
best_hyperparams_enhanced_cnn_cifar = {}
best_accuracy_deep_cnn_cifar = 0
best_hyperparams_deep_cnn_cifar = {}

print("For CNN evaluations on CIFAR-10:")
print()

for learning_rate, batch_size, optimizer_name, dropout_rate in sampled_combinations:
    
    cifar_train_loader, cifar_validation_loader = create_data_loaders(cifar_train_dataset, cifar_val_dataset, batch_size)


    # Baseline CNN for CIFAR-10
    baseline_cnn_cifar = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input channels: 3 (RGB), Output channels: 32
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),  

        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Input channels: 32, Output channels: 64
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),  

        torch.nn.Flatten(),  # Flatten the output for the fully connected layer
        torch.nn.Linear(64 * 8 * 8, 128),  # input images are 32x32
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)  # Output layer for 10 classes
    )

    # Baseline CNN for CIFAR-10
    baseline_cnn_cifar.to(device)

    time_start = time.time()
    accuracy_baseline_cnn_cifar, epoch = training_and_validation(
        model=baseline_cnn_cifar,
        train_loader=cifar_train_loader,
        val_loader=cifar_validation_loader,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name
    )
    time_end = time.time()
    print(f"CNN (baseline, 2 conv): Learning Rate: {learning_rate}, Batch Size: {batch_size}, Optimizer: {optimizer_name}, Dropout: {dropout_rate}, Accuracy: {accuracy_baseline_cnn_cifar:.2f}%, Time: {(time_end - time_start) / 60:.2f} minutes, Epoch: {epoch}")

    
    if accuracy_baseline_cnn_cifar > best_accuracy_baseline_cnn_cifar:
        best_accuracy_baseline_cnn_cifar = accuracy_baseline_cnn_cifar
        best_hyperparams_baseline_cnn_cifar = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizer': optimizer_name,
            'dropout_rate': dropout_rate
        }


    # Enhanced CNN for CIFAR-10
    enhanced_cnn_cifar = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Flatten(),
        torch.nn.Linear(64 * 8 * 8, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(128, 10)
    )

    # Enhanced CNN for CIFAR-10
    enhanced_cnn_cifar.to(device)

    time_start = time.time()
    accuracy_enhanced_cnn_cifar, epoch = training_and_validation(
        model=enhanced_cnn_cifar,
        train_loader=cifar_train_loader,
        val_loader=cifar_validation_loader,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name
    )
    time_end = time.time()
    print(f"CNN (enhanced, BN+dropout): Learning Rate: {learning_rate}, Batch Size: {batch_size}, Optimizer: {optimizer_name}, Dropout: {dropout_rate}, Accuracy: {accuracy_enhanced_cnn_cifar:.2f}%, Time: {(time_end - time_start) / 60:.2f} minutes, Epoch: {epoch}")

    if accuracy_enhanced_cnn_cifar > best_accuracy_enhanced_cnn_cifar:
        best_accuracy_enhanced_cnn_cifar = accuracy_enhanced_cnn_cifar
        best_hyperparams_enhanced_cnn_cifar = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizer': optimizer_name,
            'dropout_rate': dropout_rate
        }
    

    # Deeper CNN for CIFAR-10
    deeper_cnn_cifar = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Flatten(),
        torch.nn.Linear(128 * 4 * 4, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_rate),
        torch.nn.Linear(256, 10)
    )

    deeper_cnn_cifar.to(device)
  
    time_start = time.time()
    accuracy_deeper_cnn_cifar, epoch = training_and_validation(
        model=deeper_cnn_cifar,
        train_loader=cifar_train_loader,
        val_loader=cifar_validation_loader,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name
    )
    time_end = time.time()
    print(f"CNN (deeper, >=3 conv): Learning Rate: {learning_rate}, Batch Size: {batch_size}, Optimizer: {optimizer_name}, Dropout: {dropout_rate}, Accuracy: {accuracy_deeper_cnn_cifar:.2f}%, Time: {(time_end - time_start) / 60:.2f} minutes, Epoch: {epoch}")

    if accuracy_deeper_cnn_cifar > best_accuracy_deep_cnn_cifar:
        best_accuracy_deep_cnn_cifar = accuracy_deeper_cnn_cifar
        best_hyperparams_deep_cnn_cifar = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizer': optimizer_name,
            'dropout_rate': dropout_rate
        }
    print()

print(f"Completed CNN evaluations on CIFAR-10.")
print(f"Best hyperparameters for Baseline CNN CIFAR-10 - Batch Size: {best_hyperparams_baseline_cnn_cifar['batch_size']}, Learning Rate: {best_hyperparams_baseline_cnn_cifar['learning_rate']}, Optimizer: {best_hyperparams_baseline_cnn_cifar['optimizer']}, Dropout: {best_hyperparams_baseline_cnn_cifar['dropout_rate']}")
print(f"Best hyperparameters for Enhanced CNN CIFAR-10 - Batch Size: {best_hyperparams_enhanced_cnn_cifar['batch_size']}, Learning Rate: {best_hyperparams_enhanced_cnn_cifar['learning_rate']}, Optimizer: {best_hyperparams_enhanced_cnn_cifar['optimizer']}, Dropout: {best_hyperparams_enhanced_cnn_cifar['dropout_rate']}")
print(f"Best hyperparameters for Deeper CNN CIFAR-10 - Batch Size: {best_hyperparams_deep_cnn_cifar['batch_size']}, Learning Rate: {best_hyperparams_deep_cnn_cifar['learning_rate']}, Optimizer: {best_hyperparams_deep_cnn_cifar['optimizer']}, Dropout: {best_hyperparams_deep_cnn_cifar['dropout_rate']}")
print()


# combine train and validation sets for final training
cifar_full_train_dataset_cnn = torch.utils.data.ConcatDataset([cifar_train_dataset, cifar_val_dataset])
# Do the final evaluation on the test set for CIFAR-10 using the best hyperparameters for each architecture.

#### For baseline CNN
final_cifar_train_loader_baseline_cnn = torch.utils.data.DataLoader(cifar_full_train_dataset_cnn, batch_size=best_hyperparams_baseline_cnn_cifar['batch_size'], shuffle=True)

final_cifar_test_loader_baseline_cnn = torch.utils.data.DataLoader(cifar_test_dataset, batch_size=best_hyperparams_baseline_cnn_cifar['batch_size'], shuffle=False)
# Baseline CNN for CIFAR-10
baseline_cnn_cifar = torch.nn.Sequential(
    torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input channels: 3 (RGB), Output channels: 32
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),  

    torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Input channels: 32, Output channels: 64
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),  

    torch.nn.Flatten(),  # Flatten the output for the fully connected layer
    torch.nn.Linear(64 * 8 * 8, 128),  # input images are 32x32
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)  # Output layer for 10 classes
)

baseline_cnn_cifar.to(device)

time_start = time.time()
final_accuracy_baseline_cnn_cifar, epoch = final_training_and_testing(
        model=baseline_cnn_cifar,
        train_loader=final_cifar_train_loader_baseline_cnn,
        test_loader=final_cifar_test_loader_baseline_cnn,
        learning_rate=best_hyperparams_baseline_cnn_cifar['learning_rate'],
        optimizer_name=best_hyperparams_baseline_cnn_cifar['optimizer']
    )
time_end = time.time()
print(f"Final Test Accuracy for Baseline CNN CIFAR-10: {final_accuracy_baseline_cnn_cifar:.2f}%")
print(f"Runtime for Baseline CNN on CIFAR-10 dataset in minutes: {(time_end - time_start) / 60:.2f}")
print(f"Total Epochs for Baseline CNN on CIFAR-10: {epoch}")
print() 

### Enhanced CNN for CIFAR-10

final_cifar_train_loader_enhanced_cnn = torch.utils.data.DataLoader(cifar_full_train_dataset_cnn, batch_size=best_hyperparams_enhanced_cnn_cifar['batch_size'], shuffle=True)
# For testing
final_cifar_test_loader_enhanced_cnn = torch.utils.data.DataLoader(cifar_test_dataset, batch_size=best_hyperparams_enhanced_cnn_cifar['batch_size'], shuffle=False)

# Enhanced CNN for CIFAR-10
enhanced_cnn_cifar = torch.nn.Sequential(
    torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(32),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Dropout(best_hyperparams_enhanced_cnn_cifar['dropout_rate']),
    torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(64),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Dropout(best_hyperparams_enhanced_cnn_cifar['dropout_rate']),
    torch.nn.Flatten(),
    torch.nn.Linear(64 * 8 * 8, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(best_hyperparams_enhanced_cnn_cifar['dropout_rate']),
    torch.nn.Linear(128, 10)
)

enhanced_cnn_cifar.to(device)

time_start = time.time()
final_accuracy_enhanced_cnn_cifar, epoch = final_training_and_testing(
        model=enhanced_cnn_cifar,
        train_loader=final_cifar_train_loader_enhanced_cnn,
        test_loader=final_cifar_test_loader_enhanced_cnn,
        learning_rate=best_hyperparams_enhanced_cnn_cifar['learning_rate'],
        optimizer_name=best_hyperparams_enhanced_cnn_cifar['optimizer']
    )
time_end = time.time()
print(f"Final Test Accuracy for Enhanced CNN CIFAR-10: {final_accuracy_enhanced_cnn_cifar:.2f}%")
print(f"Runtime for Enhanced CNN on CIFAR-10 dataset in minutes: {(time_end - time_start) / 60:.2f}")
print(f"Total Epochs for Enhanced CNN on CIFAR-10: {epoch}")
print()

##### Deep CNN for CIFAR-10

final_cifar_train_loader_deep_cnn = torch.utils.data.DataLoader(cifar_full_train_dataset_cnn, batch_size=best_hyperparams_deep_cnn_cifar['batch_size'], shuffle=True)
# For testing
final_cifar_test_loader_deep_cnn = torch.utils.data.DataLoader(cifar_test_dataset, batch_size=best_hyperparams_deep_cnn_cifar['batch_size'], shuffle=False)


# Deeper CNN for CIFAR-10
deeper_cnn_cifar = torch.nn.Sequential(
    torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(32),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Dropout(best_hyperparams_deep_cnn_cifar['dropout_rate']),
    torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(64),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Dropout(best_hyperparams_deep_cnn_cifar['dropout_rate']),
    torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(128),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Dropout(best_hyperparams_deep_cnn_cifar['dropout_rate']),
    torch.nn.Flatten(),
    torch.nn.Linear(128 * 4 * 4, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(best_hyperparams_deep_cnn_cifar['dropout_rate']),
    torch.nn.Linear(256, 10)
)

deeper_cnn_cifar.to(device)

time_start = time.time()
final_accuracy_deep_cnn_cifar, epoch = final_training_and_testing(
        model=deeper_cnn_cifar,
        train_loader=final_cifar_train_loader_deep_cnn,
        test_loader=final_cifar_test_loader_deep_cnn,
        learning_rate=best_hyperparams_deep_cnn_cifar['learning_rate'],
        optimizer_name=best_hyperparams_deep_cnn_cifar['optimizer']
    )
time_end = time.time()
print(f"Final Test Accuracy for Deep CNN CIFAR-10: {final_accuracy_deep_cnn_cifar:.2f}%")
print(f"Runtime for Deep CNN on CIFAR-10 dataset in minutes: {(time_end - time_start) / 60:.2f}")
print(f"Total Epochs for Deep CNN on CIFAR-10: {epoch}")
print()

