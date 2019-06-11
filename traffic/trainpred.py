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
from preprocess import preprocess

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



def loss_function(pred, data_y):
    """ loss function """
    L2_loss = F.mse_loss(pred, data_y, size_average=False)
    return L2_loss

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
    with torch.no_grad():
        for data in test_loader:
            data, data_tp, data_y = [d.to(device, dtype=torch.float ) for d in data ]
            pred_map = model(data , data_tp)
            test_loss += loss_function(pred_map , data_y)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


# Main training
if __name__ == "__main__": 

    args = parser.parse_args()

    torch.manual_seed(123)
    cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if cuda else "cpu")

    LOOKBACK = 8
    LOOKFORWARD = 1
    DSHAPE = [46,141]

    dataset_train = MapDataset('demand_map.npy', 'tp_data.npy', LOOKBACK, LOOKFORWARD,
                                            train=True)
    dataset_test = MapDataset('demand_map.npy', 'tp_data.npy', LOOKBACK, LOOKFORWARD,
                                            train=False)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=2)


    model = PRED( LOOKBACK, LOOKFORWARD, DSHAPE ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    earlystopping = EarlyStopping('min', patience=30)


    # check vae dir exists, if not, create it
    save_dir = join(args.logdir, 'vae')
    if not exists(save_dir):
        mkdir(save_dir)
        mkdir(join(save_dir, 'samples'))

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

    if (args.prediction) :
        
        # df = preprocess(data_path)
        data, data_tp, data_y = [  d.to(device, dtype=torch.float ) for d in next(iter(test_loader)) ]
        model.eval()
        pred = model(data, data_tp)
        
        show_plot = True
        if (show_plot) :

            import seaborn as sns
            import matplotlib.pylab as plt
            print(pred.shape)
            pred = pred.cpu().detach().numpy()
            data = data.cpu().detach().numpy()

            fig, ax = plt.subplots(2,2)
            for i in range(2):
                for j in range(2):
                    ax[i, j].text(0.5, 0.5, str((i, j)),
                                fontsize=18, ha='center')
            sns.heatmap(data[2][0], linewidth=0.5 , cmap="BuPu" , ax = ax[0,0]) 
            sns.heatmap(pred[1][0], linewidth=0.5 , cmap="BuPu" , ax = ax[0,1]) 
            sns.heatmap(data[1][0], linewidth=0.5 , cmap="BuPu" , ax = ax[1,0]) 
            sns.heatmap(pred[0][0], linewidth=0.5 , cmap="BuPu" , ax = ax[1,1]) 
            plt.show()


    else :
        cur_best = None
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
                break


