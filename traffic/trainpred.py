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
# from utils.misc import LSIZE, RED_SIZE

from utils import EarlyStopping
# from utils.learning import ReduceLROnPlateau

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

def train(epoch, model, train_loader):
    """ One training epoch """
    model.train()
    # dataset_train.load_next_buffer()
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
    # dataset_test.load_next_buffer()
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
                        help='specify prediction mode')

    parser.add_argument('--data_path', type=str, 
                        help='specify prediction mode')

    parser.add_argument('--lookback', type=int, default=8, 
                        help='Step look back (default: 8)')
    parser.add_argument('--lookforward', type=int, default=5, 
                        help='Step look forward (default: 5)')


    args = parser.parse_args()

    torch.manual_seed(123)
    cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if cuda else "cpu")

    LOOKBACK = args.lookback
    LOOKFORWARD = args.lookforward


    # if input is str load config
    tclass_path = join(args.logdir, 'tclass.pkl')
    openfile = open(tclass_path , 'rb')
    tclass = pickle.load(openfile)
    print (tclass.data )

    npdata = tclass.data['demand_map'].values
    npdata = np.array(list(npdata), dtype= np.float)

    tp_data = tclass.data[ ['day', 'timestamp']].values
    tp_data = np.array(tp_data, dtype= np.float)

    DSHAPE = tclass.data_shape

    # Model Instantiate 
    model = PRED( LOOKBACK, LOOKFORWARD, DSHAPE ).to(device)
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



    dataset_train = MapDataset( npdata , tp_data, LOOKBACK, LOOKFORWARD,
                                            train=True)
    dataset_test = MapDataset( npdata, tp_data, LOOKBACK, LOOKFORWARD,
                                            train=False)
    
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=2)


    if (args.prediction) :    
        # df = tclass.preprocess(data_path)
        # data_path = './traffic-management/Traffic Management/training.csv'
        data = pd.read_csv( args.data_path )
        data = tclass.geohashToGeocoord(data)
        data = tclass.preprocess(data)
        
        npdata = tclass.data['demand_map'].values
        npdata = np.array(list(npdata), dtype= np.float)
        npdata = torch.tensor(npdata[-LOOKBACK:] ).to( device, dtype=torch.float ).unsqueeze(0)

        tp_data = tclass.data[ ['day', 'timestamp']].values
        tp_data = np.array(tp_data, dtype= np.float)
        tp_data = torch.tensor(tp_data[-LOOKBACK:] ).to( device, dtype=torch.float ).unsqueeze(0) 
        
        model.eval()
        pred = model ( npdata, tp_data)
        
        pred = pred.cpu().detach().numpy()

        print(pred)

    if (args.test) :        
        # df = tclass.preprocess(data_path)
        # pred_dat_path = './traffic-management/Traffic Management/training.csv'
        data = pd.read_csv( args.data_path )
        data = tclass.geohashToGeocoord(data)
        data = tclass.preprocess(data)
        
        npdata = tclass.data['demand_map'].values
        npdata = np.array(list(npdata), dtype= np.float)

        tp_data = tclass.data[ ['day', 'timestamp']].values
        tp_data = np.array(tp_data, dtype= np.float)
        
        pred_dataset = MapDataset( npdata, tp_data, LOOKBACK, LOOKFORWARD,
                                                train=False)
        pred_loader = torch.utils.data.DataLoader(
                pred_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        test(model, pred_loader)
        # dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
        # data, data_tp, data_y = [  d.to(device, dtype=torch.float ) for d in next(iter(pred_loader)) ]
        # model.eval()
        # pred = model(data, data_tp)
        # MAE = l1_function(pred, data_y) 
        # MSE = loss_function(pred, data_y)

        # print(f' MAE : {MAE}\n MSE : {MSE}')

        # show_plot = True
        # if (show_plot) :

        #     import seaborn as sns
        #     import matplotlib.pylab as plt
        #     print(pred.shape)
        #     pred = pred.cpu().detach().numpy()
        #     data = data.cpu().detach().numpy()

        #     fig, ax = plt.subplots(2,2)
        #     for i in range(2):
        #         for j in range(2):
        #             ax[i, j].text(0.5, 0.5, str((i, j)),
        #                         fontsize=18, ha='center')
        #     sns.heatmap(data[2][-1], linewidth=0.5 , cmap="BuPu" , ax = ax[0,0]) 
        #     sns.heatmap(pred[1][0], linewidth=0.5 , cmap="BuPu" , ax = ax[0,1]) 
        #     sns.heatmap(data[5][-1], linewidth=0.5 , cmap="BuPu" , ax = ax[1,0]) 
        #     sns.heatmap(pred[0][4], linewidth=0.5 , cmap="BuPu" , ax = ax[1,1]) 
        #     plt.show()


    else :
        cur_best = None
        best_epoch = None
        for epoch in range(1, args.epochs + 1):
            train(epoch, model, train_loader)
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


