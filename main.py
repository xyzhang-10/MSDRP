import os
import sys
sys.path.append("..")
import torch.multiprocessing
import argparse
import torch.utils.data
from network import MSDRP
import fitlog
from utils import *

def main():
    # Training settings
    parser = argparse.ArgumentParser(description = 'Model')
    parser.add_argument('--epochs', type = int, default =200,
                        metavar = 'N', help = 'number of epochs to train')
    parser.add_argument('--lr', type = float, default = 0.0001,
                        metavar = 'FLOAT', help = 'learning rate')
    parser.add_argument('--embed_dim', type = int, default = 128,
                        metavar = 'N', help = 'embedding dimension')
    parser.add_argument('--weight_decay', type = float, default = 0.0003,
                        metavar = 'FLOAT', help = 'weight decay')
    parser.add_argument('--droprate', type = float, default = 0.5,
                        metavar = 'FLOAT', help = 'dropout rate')
    parser.add_argument('--batch_size', type = int, default =64,
                        metavar = 'N', help = 'input batch size for training')
    parser.add_argument('--test_batch_size', type = int, default =64,
                        metavar = 'N', help = 'input batch size for testing')
    parser.add_argument('--rawpath', type=str, default='data/',
                        metavar='STRING', help='rawpath')
    parser.add_argument('--device', type=str, default='cuda:3',
                        help='device')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for earlystopping (default: 10)')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or test')
    parser.add_argument('--weight_path', type=str, default='best2',
                        help='filepath for pretrained weights')
    args = parser.parse_args()

    print('-------------------- Hyperparams --------------------')
    print('weight decay: ' + str(args.weight_decay))
    print('dropout rate: ' + str(args.droprate))
    print('learning rate: ' + str(args.lr))
    print('dimension of embedding: ' + str(args.embed_dim))

    train_test_model(args)

def train_test_model(args):
    train_loader, test_loader, val_loader = load_data(args)
    model = MSDRP(2040, 2698,args.embed_dim, args.batch_size, args.droprate, args.droprate).to(args.device)
    if args.mode == 'train':
        Regression_criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        log_folder = os.path.join(os.getcwd(), "result/log_gene_best2", model._get_name())
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        fitlog.set_log_dir(log_folder)
        fitlog.add_hyper(args)
        fitlog.add_hyper_in_file(__file__)

        stopper = EarlyStopping(mode='lower', patience=args.patience)
        for epoch in range(1, args.epochs + 1):
            print("=====Epoch {}".format(epoch))
            print("Training...")
            train_loss = train(model,train_loader, Regression_criterion, optimizer, args.device)
            fitlog.add_loss(train_loss.item(), name='Train MSE', step=epoch)


            fitlog.add_loss(train_loss.item(), name='Train MSE', step=epoch)
            print('Evaluating...')
            rmse, _, _, _ = validate(model, val_loader, args.device)
            print("Validation rmse:{}".format(rmse))
            fitlog.add_metric({'val': {'RMSE': rmse}}, step=epoch)

            early_stop = stopper.step(rmse, model)
            if early_stop:
                break

        print('EarlyStopping! Finish training!')
        print('Testing...')
        stopper.load_checkpoint(model)
        torch.save(model.state_dict(), 'weight/{}.pth'.format(args.weight_path))
        train_rmse, train_MAE, train_r2, train_r = validate(model, train_loader, args.device)
        val_rmse, val_MAE, val_r2, val_r = validate(model, val_loader, args.device)
        test_rmse, test_MAE, test_r2, test_r = validate(model, test_loader, args.device)
        print('Train reslut: rmse:{} mae:{} r:{}'.format(train_rmse,train_MAE, train_r))
        print('Val reslut: rmse:{} mae:{} r:{}'.format(val_rmse, val_MAE, val_r))
        print('Test reslut: rmse:{} mae:{}  r:{}'.format(test_rmse, test_MAE, test_r))

        fitlog.add_best_metric(
            {'epoch': epoch - args.patience,
                "train": {'RMSE': train_rmse, 'MAE': train_MAE, 'pearson': train_r, "R2": train_r2},
                "valid": {'RMSE': stopper.best_score, 'MAE': val_MAE, 'pearson': val_r, 'R2': val_r2},
                "test": {'RMSE': test_rmse, 'MAE': test_MAE, 'pearson': test_r, 'R2': test_r2}})

    elif args.mode == 'test':
        weight = ""
        model.load_state_dict(
            torch.load('weight/{}.pth'.format(args.weight_path), map_location=args.device)['model_state_dict'])
        test_rmse, test_MAE, test_r2, test_r = validate(model, test_loader, args.device)
        print('Test RMSE: {}, MAE: {}, R2: {}, R: {}'.format(round(test_rmse.item(), 4), round(test_MAE, 4),
                                                         round(test_r2, 4), round(test_r, 4)))





if __name__ == "__main__":
    main()
    print('hello zxy log_gene_best2')