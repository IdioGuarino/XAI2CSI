import torch
from torch import nn


class INET(nn.Module):   
    def __init__(self, dataset, num_classes, output_h=8, output_w=8, **kwargs):
        super(INET, self).__init__()

        self.dataset = dataset

        self.antennas = kwargs.get('antennas', 4)
        self.subcarriers = kwargs.get('subcarriers', 64)
        self.window_size = kwargs.get('window_size', 51)
        self.out_features_size = kwargs.get('out_features_size', 128)

        self.output_h = output_h
        self.output_w = output_w

        # self.branch_1_out=16 #era 5
        # self.branch_2_0_out=16 #era 3
        # self.branch_2_1_out=16 #era 6
        # self.branch_2_2_out=16 #era 9

        self.branch_1_out = 8
        self.branch_2_0_out = 8
        self.branch_2_1_out = 8
        self.branch_2_2_out = 8
        self.conv3_out_channels = 16

        # self.conv3_out_channels = 32 #output_channels era 3
        self.mpool0 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.adpool0 = nn.AdaptiveMaxPool2d((self.output_h, self.output_w))
        
        self.conv1_0 = nn.Conv2d(self.antennas, self.branch_1_out, (2, 2), stride=(2,2), padding=self._get_padding('valid', (2,2)))
        self.bn1_0 = nn.BatchNorm2d(self.branch_1_out)
        self.relu1_0 = nn.ReLU()
        self.adpool1 = nn.AdaptiveMaxPool2d((self.output_h, self.output_w))


        self.conv2_0 = nn.Conv2d(self.antennas, self.branch_2_0_out, (1, 1), stride=(1,1), padding=self._get_padding('same', (1,1)))
        self.bn2_0 = nn.BatchNorm2d(self.branch_2_0_out)
        self.relu2_0 = nn.ReLU()

        self.conv2_1 = nn.Conv2d(self.branch_2_0_out, self.branch_2_1_out, (2, 2), stride=(1,1), padding=self._get_padding('same', (2,2)))
        self.bn2_1 = nn.BatchNorm2d(self.branch_2_1_out)
        self.relu2_1 = nn.ReLU()

        self.conv2_2 = nn.Conv2d(self.branch_2_1_out, self.branch_2_2_out, (4, 4), stride=(2,2), padding=self._get_padding('same', (4,4)))
        self.bn2_2 = nn.BatchNorm2d(self.branch_2_2_out)
        self.relu2_2 = nn.ReLU()
        self.adpool2 = nn.AdaptiveMaxPool2d((self.output_h, self.output_w))

        
        # self.conv4 = Conv2d_BN(in_channels=self.antennas+self.branch_1_out+self.branch_2_2_out, out_channels=out_channels_c4, kernel_size=(1, 1))
        self.conv3=nn.Conv2d(
                            in_channels=self.antennas+self.branch_1_out+self.branch_2_2_out, 
                            out_channels=self.conv3_out_channels, 
                            kernel_size=(1, 1),
                            stride=(1,1),
                            padding=self._get_padding('same', (1, 1))
                            )
        self.bn3 = nn.BatchNorm2d(self.conv3_out_channels)
        self.relu3 = nn.ReLU()

        # Dense layers
        self.input_size = self._get_flattened_dim(self.conv3_out_channels, output_h, output_w)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(self.input_size, self.out_features_size)
        self.relu_fc1 = nn.ReLU()
        self.fc = nn.Linear(self.out_features_size, num_classes)
        # self.fc = nn.Linear(self.input_size, num_classes)
        self.head_var = 'fc'

    def _get_padding(self, padding_type, kernel_size):
        if padding_type == 'same':
            if isinstance(kernel_size, int):
                return kernel_size // 2
            else:
                return (kernel_size[0] // 2, kernel_size[1] // 2)
        return 0
    
    def _get_flattened_dim(self, pre_out_channels,output_h, output_w):
        # Dynamically calculate the flattened dimension based on the output size of the last convolutional layer
        return pre_out_channels * output_h * output_w #

    def extract_features(self, x):

        x0 = self.mpool0(x)
        x0 = self.adpool0(x0)

        x1 = self.conv1_0(x)
        x1 = self.bn1_0(x1)
        x1 = self.relu1_0(x1)
        x1 = self.adpool1(x1)

        x2 = self.conv2_0(x)
        x2 = self.bn2_0(x2)
        x2 = self.relu2_0(x2)

        x2 = self.conv2_1(x2)
        x2 = self.bn2_1(x2)
        x2 = self.relu2_1(x2)

        x2 = self.conv2_2(x2)
        x2 = self.bn2_2(x2)
        x2 = self.relu2_2(x2)
        x2 = self.adpool2(x2)

        x_concat = torch.cat([x0, x1, x2], dim=1)
        # print(x_concat.shape)
        out = self.conv3(x_concat)
        out = self.bn3(out)
        out = self.relu3(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu_fc1(out)
        out = self.dropout(out)
        return out

    def forward(self, x):
        
        out = self.extract_features(x)
        out = self.fc(out)
        return out