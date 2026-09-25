from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import os

NUM_WORKERS = os.cpu_count()
def create_dataloaders(
    
    train_dir,
    test_dir,
    batch_size,
    train_transform,
    test_transform,
    num_workers=NUM_WORKERS):
    """
    args: 
        train_dir -> train data path
        test_dir -> test data path
        batch_size -> batch size
        train_transform -> train data transform
        test_transform -> test data transform
        
    return:
       returns train_loader, test_loader , validation_loader

    """
    
    
    test_data = ImageFolder(train_dir, transform=test_transform)
    train_dataset = ImageFolder(test_dir, transform=train_transform)
    test_dataset, validation_dataset = random_split(test_data, [272, 100])
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True )
    validation_loader = DataLoader(validation_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader, validation_loader
    
    
   
    
    