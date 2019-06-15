
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PRED(nn.Module): 
    """  pred-coder """
    def __init__(self, look_back, look_forward , data_shape, concatTime=False):
        super(PRED, self).__init__()

        self.concatTime = concatTime

        self.channel_in = look_back + 2  if concatTime else look_back
        self.look_forward = look_forward
        self.data_shape = data_shape
        self.total_node = np.prod (data_shape)

        
        self.conv1 = nn.Conv2d(self.channel_in , 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 2 , padding=1)
        self.conv3 = nn.Conv2d(64, 128, 2 , padding=1)
        self.conv4 = nn.Conv2d(128, 256, 2 , padding=1)
        # self.conv5 = nn.Conv2d(256, 512, 2 , padding=1)

        self.pool2d_2 = nn.MaxPool2d(2)


        self.dense = nn.Linear( 2304 , self.total_node   )
        # self.dense2 = nn.Linear( self.total_node , self.look_forward * self.total_node   )

        self.convOut = nn.Conv2d( 1, look_forward, 3 , padding=1)
        self.convOut2 = nn.Conv2d( look_forward, look_forward, 3 , padding=1)

    def forward(self, x, x1): 

        if self.concatTime : 
            # x1 = F.pad(x1,(0,139))
            x1 = x1.unsqueeze(-1).repeat(1,1, 1,self.total_node)
            x1 = x1.reshape( x1.size(0), -1, self.data_shape[-2], self.data_shape[-1] )

            x = torch.cat( (x,x1), dim =-3)

        x1 = x1.view(x1.size(0),-1)

        x = F.relu(self.conv1(x))
        x = self.pool2d_2(x)

        x = F.relu(self.conv2(x))
        x = self.pool2d_2(x)

        x = F.relu(self.conv3(x))
        x = self.pool2d_2(x)

        x = F.relu(self.conv4(x))
        x = self.pool2d_2(x)

        # x = F.relu(self.conv5(x))
        # x = self.pool2d_2(x)

        x = x.view(x.size(0), -1)
        
        out = self.dense(x)
        # print(out.size())
        # out =  self.dense2(out)
        out = out.reshape ( x.size(0), 1, self.data_shape[-2], self.data_shape[-1] )
        # out = F.relu(self.deconv4(out))
        # out = F.relu(self.deconv5(out))
        out = (self.convOut(out))
        out = F.relu(self.convOut2(out))

        return out

