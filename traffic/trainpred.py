""" Training VAE """
import argparse
from os.path import join, exists
from os import mkdir

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.pred import PRED

from utils import save_checkpoint
from utils import EarlyStopping

from dataloader import MapDataset
from dataprocess import Traffic
import pickle

import numpy as np
import pandas as pd

def loss_function(pred, data_y):
    """ loss function """
    L2_loss = F.mse_loss(pred, data_y, size_average=False)
    return L2_loss

def l1_function(pred, data_y):
    """ loss function """
    L1_loss = F.l1_loss(pred, data_y, size_average=False)
    return L1_loss

def train(epoch, model, train_loader, optimizer ):
    """ One training epoch """
    model.train()
    device = next(model.parameters()).device
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data_x, data_tp, data_y =[ d.to(device, dtype=torch.float ) for d in data ]
        optimizer.zero_grad()
        pred_map = model(data_x , data_tp)

        loss = loss_function(pred_map , data_y)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data_x)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(model, test_loader):
    """ One test epoch """
    model.eval()
    device = next(model.parameters()).device
    test_loss = 0
    test_l1   = 0
    with torch.no_grad():
        for data in test_loader:
            data, data_tp, data_y = [d.to(device, dtype=torch.float ) for d in data ]
            pred_map = model(data , data_tp)
            test_loss += loss_function(pred_map , data_y)
            test_l1 += l1_function(pred_map , data_y)

    test_loss /= len(test_loader.dataset)
    test_l1/= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('====> Test set MAE: {:.4f}'.format(test_l1))

    #train with l2 loss only
    return test_loss



class TrainTraffic () :
    def __init__(self, args) :
        self.args = args 

        torch.manual_seed(123)
        cuda = torch.cuda.is_available()
        torch.backends.cudnn.benchmark = True

        self.device = torch.device("cuda" if cuda else "cpu")

        self.LOOKBACK = args.lookback
        self.LOOKFORWARD = args.lookforward
        
        filehandler = open(join(args.logdir, 'tclass.pkl') , 'rb')
        self.tclass = pickle.load(filehandler)
        
        self.DSHAPE = self.tclass.data_shape
        print(self.DSHAPE)
        # Model Instantiate 
        self.model = PRED( self.LOOKBACK, self.LOOKFORWARD, self.DSHAPE ).to(self.device)


    def load_model(self, model, logdir ) :
        # Load model routine
        save_dir = join(logdir, 'model')
          
        reload_file = join(save_dir, 'best.tar')
        if exists(reload_file):
            state = torch.load(reload_file)
            print("Reloading model at epoch {}"
                ", with test error {}".format(
                    state['epoch'],
                    state['precision']))
            model.load_state_dict(state['state_dict'])
        return model

    def prediction (self, data_path, device='cpu' ) :   
        
        model = self.model.to(device)
        self.load_model(model, self.args.logdir )

        data = pd.read_csv(data_path, index_col=None, header=0)
        data = self.tclass.geohash_decode(data)
        data = self.tclass.preprocess(data)

        # data = self.tclass.saved_data

        npdata = data['demand_map'].values
        npdata = np.array(list(npdata), dtype= np.float)
        npdata = torch.tensor(npdata[-self.LOOKBACK:] ).to( device, dtype=torch.float ).unsqueeze(0)

        tp_data = data[ ['day', 'timestamp']].values
        tp_data = np.array(tp_data, dtype= np.float)
        tp_data = torch.tensor(tp_data[-self.LOOKBACK:] ).to( device, dtype=torch.float ).unsqueeze(0) 
        
        model.eval()
        pred = model ( npdata, tp_data)
        
        pred = pred.cpu().detach().numpy()
        return pred

        
    def test_benchmark (self, data_path):
        # benchmark model with new data

        model = self.model.to(self.device)
        model = self.load_model ( model, self.args.logdir )

        data = pd.read_csv(data_path, index_col=None, header=0)

        data = self.tclass.geohash_decode(data)
        data = self.tclass.preprocess(data)

        # data = self.tclass.saved_data
        
        npdata = data['demand_map'].values
        npdata = np.array(list(npdata), dtype= np.float)

        tp_data = data[ ['day', 'timestamp']].values
        tp_data = np.array(tp_data, dtype= np.float)
        
        test_dataset = MapDataset( npdata, tp_data, self.LOOKBACK, self.LOOKFORWARD,
                                                train=False, ratio=0)
        test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        test(model, test_loader)

    def train_model(self):
        args = self.args

        model = self.model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
        earlystopping = EarlyStopping('min', patience=30)

        # check vae dir exists, if not, create it
        save_dir = join(args.logdir, 'model')
        if not exists(save_dir):
            mkdir(save_dir)

        reload_file = join(save_dir, 'best.tar')
        if not args.noreload and exists(reload_file):
            state = torch.load(reload_file)
            print("Reloading model at epoch {}"
                ", with test error {}".format(
                    state['epoch'],
                    state['precision']))
            model.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            earlystopping.load_state_dict(state['earlystopping'])

        # retrain with new dataset with same map dimension
        if (args.data_path) :
            print('TRAIN with new dataset within same geohash range ')
            data = pd.read_csv( args.data_path )
            data = self.tclass.geohash_decode(data)
            data = self.tclass.preprocess(data)
        else :
            data = self.tclass.saved_data 

        npdata = data['demand_map'].values
        npdata = np.array(list(npdata), dtype= np.float)

        tp_data = data[ ['day', 'timestamp']].values
        tp_data = np.array(tp_data, dtype= np.float)

        dataset_train = MapDataset( npdata , tp_data, self.LOOKBACK, self.LOOKFORWARD,
                                                train=True)
        dataset_test = MapDataset( npdata, tp_data, self.LOOKBACK, self.LOOKFORWARD,
                                                train=False)
        
        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=2)

        cur_best = None
        best_epoch = None
        for epoch in range(1, args.epochs + 1):
            train(epoch, model, train_loader, optimizer)
            test_loss = test(model, test_loader)
            scheduler.step(test_loss)
            earlystopping.step(test_loss)

            # checkpointing
            best_filename = join(save_dir, 'best.tar')
            filename = join(save_dir, 'checkpoint.tar')
            is_best = not cur_best or test_loss < cur_best
            if is_best:
                cur_best = test_loss
                best_epoch = epoch

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'precision': test_loss,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'earlystopping': earlystopping.state_dict()
            }, is_best, filename, best_filename)

            if earlystopping.stop:
                print("End of Training because of early stopping at epoch {}".format(epoch))
                print(f"Best epoch {best_epoch} with loss :{cur_best}")
                break




# Main training
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--logdir', type=str, help='Directory where results are logged')
    parser.add_argument('--noreload', action='store_true',
                        help='Best model is not reloaded if specified')
    parser.add_argument('--prediction', action='store_true',
                        help='specify prediction mode')
    parser.add_argument('--test', action='store_true',
                        help='specify test mode')

    parser.add_argument('--data_path', type=str, 
                        help='path to data (csv) ')

    parser.add_argument('--lookback', type=int, default=8, 
                        help='Step look back (default: 8)')
    parser.add_argument('--lookforward', type=int, default=5, 
                        help='Step look forward (default: 5)')


    args = parser.parse_args()

    agent = TrainTraffic(args)

    if (args.prediction) :
        pred = agent.prediction( args.data_path)
        print(pred.shape)
        pred_df = agent.tclass.postprocess(np.squeeze(pred, 0) )
        print(pred_df)
        # Post process

    elif (args.test) :
        agent.test_benchmark( args.data_path)
    else :
        agent.train_model()

