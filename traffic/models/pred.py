
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PRED(nn.Module): # pylint: disable=too-many-instance-attributes
    """ VAE encoder """
    def __init__(self, look_back, look_forward , data_shape):
        super(PRED, self).__init__()
        #FIXME
        look_forward = 1
        self.look_back = look_back
        self.look_forward = look_forward
        self.data_shape = data_shape
        
        # appended_data_shape = np.add(data_shape , [0,1] )

        # print(appended_data_shape)
        
        # d_size = np.prod(appended_data_shape ) 
        # print(d_size)
        
        self.conv1 = nn.Conv2d(look_back, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 2 , padding=1)
        self.conv3 = nn.Conv2d(64, 128, 2 , padding=1)
        self.conv4 = nn.Conv2d(128, 256, 2 , padding=1)
        self.conv5 = nn.Conv2d(256, 512, 2 , padding=1)

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.pool5 = nn.MaxPool2d(2)


        # dense_size = int(d_size/ 16 )
        # self.dense1 = nn.Linear( 64 * dense_size , look_forward * d_size  ) 26899
        # self.dense1 = nn.Linear( 26880 , 6800  ) 6912
        self.dense2 = nn.Linear( 5120 , np.prod (data_shape)  )
        self.deconv4 = nn.ConvTranspose2d( 1 , look_forward, 3, padding=1)
        self.deconv5 = nn.ConvTranspose2d( 1 , look_forward, 3, padding=1)


        self.convOut = nn.Conv2d( 1, look_forward, 3 , padding=1)

    def forward(self, x, x1): # pylint: disable=arguments-differ

#       target = torch.zeros(30, 35, 512)
#       source = torch.ones(30, 35, 49)
        x1 = F.pad(x1,(0,139))
        # print(x1.size())
        # print(x.size())

        x = torch.cat( (x,x1), dim =-2)

        x1 = x1.view(x1.size(0),-1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.pool1(x))

        x = F.relu(self.conv2(x))
        x = F.relu(self.pool2(x))

        x = F.relu(self.conv3(x))
        x = F.relu(self.pool3(x))

        x = F.relu(self.conv4(x))
        x = F.relu(self.pool4(x))

        x = F.relu(self.conv5(x))
        x = F.relu(self.pool5(x))

        x = x.view(x.size(0), -1)
        # print(x.size())
        # print(x1.size())

        # x = torch.cat( (x,x1), dim =-1)
        # out = self.dense1(x)
        out = self.dense2(x)
        # print(out.size())
        out = out.reshape ( x.size(0), 1, self.data_shape[-2], self.data_shape[-1] )
        # out = F.relu(self.deconv4(out))
        # out = F.relu(self.deconv5(out))
        out = self.convOut(out)

        return out

