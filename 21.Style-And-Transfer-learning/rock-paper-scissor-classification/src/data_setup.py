from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

def create_dataloaders(
    
    train_dir,
    test_dir,
    batch_size,
    train_transform,
    test_transform,
    ):
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
    
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    test_data = ImageFolder(test_dir, transform=test_transform)
    test_dataset, validation_dataset = random_split(test_data, [272, 100])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, )
    test_loader = DataLoader(test_dataset, batch_size=batch_size//2, shuffle=False )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size//2, shuffle=False)
    
    return train_loader, test_loader, validation_loader
    
    
   
    
    