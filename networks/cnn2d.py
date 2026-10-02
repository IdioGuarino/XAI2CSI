import torch
from torch import nn
# from .network import get_output_dim

data_config={
    'cominelli23':{
        'in_channels': [32, 64],
        'out_channels': [32, 64, 96],
        'kernel_sizes': [(3, 3), (3, 3), (3, 3)],
        'strides': [(1, 1), (1, 1), (1, 1)]
    }

}

def suggest_kernel_stride(n_subcarriers):
    """
    Suggerisce tuple di (kernel_sizes, strides) per tre layer Conv2d,
    adattando le dimensioni in base al numero di subcarrier (altezza dell'input).
    """

    if n_subcarriers >= 1024:
        kernel_sizes = [(7, 3), (5, 3), (3, 3)]
        strides      = [(3, 1), (2, 1), (1, 1)]

    elif n_subcarriers >= 512:
        # kernel_sizes = [(5, 3), (3, 3), (3, 3)]
        # strides      = [(2, 1), (2, 1), (1, 1)]
        kernel_sizes = [7, (5, 3), (3, 3)]
        strides      = [(3, 1), (2, 2), 1]

    elif n_subcarriers >= 256:
        # kernel_sizes = [(5, 3), (3, 3), (3, 3)]
        # strides      = [(2, 1), (2, 1), (1, 1)]
        kernel_sizes = [7, (5, 3), (3, 3)]
        strides      = [(3, 1), (2, 2), 1]

    elif n_subcarriers >= 128:
        # kernel_sizes = [(3, 3), (3, 3), (3, 3)]
        # strides      = [(2, 1), (1, 1), (1, 1)]
        kernel_sizes = [7, (5, 3), (3, 3)]
        strides      = [(3, 1), (2, 2), 1]

    else:
        kernel_sizes = [(3, 3), (3, 3), (3, 3)]
        strides      = [(1, 1), (1, 1), (1, 1)]

    return kernel_sizes, strides
    

class CNN2D(nn.Module):   
    def __init__(self, dataset, num_classes, **kwargs):
        super(CNN2D, self).__init__()

        self.dataset = dataset

        self.antennas = kwargs.get('antennas', 4)
        self.subcarriers = kwargs.get('subcarriers', 64)
        self.window_size = kwargs.get('window_size', 51)
        self.out_features_size = kwargs.get('out_features_size', 128)

        if kwargs.get('amplitude', False) and kwargs.get('phase', False):
            self.antennas = self.antennas + self.antennas-1
        elif kwargs.get('amplitude', False):
            self.antennas = self.antennas
        else:
            self.antennas = self.antennas -1
        print("***Input shape***: ", (self.antennas, self.subcarriers, self.window_size))
        oc= data_config[dataset]['out_channels']
        ic= [self.antennas] + oc[:-1]
        print(f"***Conv layers config***: in_channels={ic}, out_channels={oc}")
        kernel_sizes, strides = suggest_kernel_stride(self.subcarriers)
        self.conv1=nn.Conv2d(
            in_channels=ic[0],
            out_channels=oc[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0]
        )

        self.mpool1=nn.MaxPool2d(kernel_size=2)
        self.conv2=nn.Conv2d(
            in_channels=ic[1],
            out_channels=oc[1],
            kernel_size=kernel_sizes[1],
            stride=strides[1],
            padding=(1,0)
        )
        self.mpool2=nn.MaxPool2d(kernel_size=2)
        self.conv3=nn.Conv2d(
            in_channels=ic[2],
            out_channels=oc[2],
            kernel_size=kernel_sizes[2],
            stride=strides[2]
        )
        self.mpool3=nn.MaxPool2d(kernel_size=2)
        
        self.drop1=nn.Dropout(0.3)
        self.drop2=nn.Dropout(0.3)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

        self.input_size = self.get_output_dim()  # Get the output dimension after convolutions and pooling
        self.fc1 = nn.Linear(self.input_size, self.out_features_size)
        self.fc = nn.Linear(self.out_features_size, num_classes)
        self.head_var = 'fc'


    def get_output_dim(self):
        
        with torch.no_grad():
            print(self.antennas, self.subcarriers, self.window_size)
            dummy_input = torch.zeros((1,self.antennas, self.subcarriers, self.window_size))  
            # Temporarily define a partial forward pass for feature extraction
            x = self.conv1(dummy_input)
            x = self.relu1(x)
            x = self.mpool1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.mpool2(x)
            x = self.conv3(x)
            x = self.relu3(x)
            x = self.mpool3(x)
            x = self.drop1(x)
            return torch.flatten(x, 1).shape[1]
    
    def forward(self, x):
        out=self.extract_features(x)
        out=self.fc(out)
        return out

    def extract_features(self, x):

        out = self.conv1(x)

        out = self.relu1(out)
        out = self.mpool1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.mpool2(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.mpool3(out)

        out = self.drop1(out)
        out = torch.flatten(out, 1)  # Flatten the tensor
        out = self.fc1(out)  # Fully connected layer
        out = self.relu4(out)
        out = self.drop2(out)
        return out