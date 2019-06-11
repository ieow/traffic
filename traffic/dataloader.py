from torch.utils.data import Dataset
import numpy as np

class MapDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, filename, tp_filename, look_back, look_forward, transform=None, train=True):
        """
        Args:
            filename        : Path to demand_map data
            tp_filename     : Path to tp data 
            look_back       : look back data used as input
            look_forward    : forward prediction step 
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.landmarks_frame = pd.read_csv(csv_file)
        ratio = 0.7
        data = np.load(filename)
        data = np.array(list(data))

        tp_data = np.load(tp_filename) 
        tp_data = np.array(list(tp_data))
        

        idx_end = int( len(data) * ratio )

        self.look_back = look_back
        self.look_forward = look_forward

        if train : 
            data = data[:idx_end]
            tp_data = tp_data[:idx_end]
        else : 
            data = data[idx_end:]
            tp_data = tp_data[idx_end:]

            
        self.data = data
        self.tp_data = np.expand_dims(tp_data, axis=1)

        self._transform = transform

    def __len__(self):
        return len(self.data) - self.look_back - self.look_forward

    def __getitem__(self, idx):
        target  = idx + self.look_back 
        
        pred    = target + self.look_forward + 1
        # print(self.data[idx:target].shape)
        x = self.data[ idx : target]
        x1 = self.tp_data[idx : target]
        y = self.data[ pred - 1 : pred ]

        # print(x.size())
        return x, x1, y
if __name__ == "__main__": 
    dataset = MapDataset('data_map.npy', 'data_tp.npy', train=True)