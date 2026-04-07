import tonic
import torchvision
from torch.utils.data import DataLoader

def get_dvs_gesture_loaders(batch_size=16, data_path='./data'):
    sensor_size = tonic.datasets.DVSGesture.sensor_size  # (128, 128, 2)
    frame_length = 50000  # 50 ms per frame
    
    transform = tonic.transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=frame_length),
        torchvision.transforms.ToTensor(),
    ])
    
    trainset = tonic.datasets.DVSGesture(save_to=data_path, train=True, transform=transform)
    testset = tonic.datasets.DVSGesture(save_to=data_path, train=False, transform=transform)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                              collate_fn=tonic.collation.PadTensors())
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                             collate_fn=tonic.collation.PadTensors())
    
    return train_loader, test_loader