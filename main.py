import torch
import torch.nn as nn
import time
import argparse
import os
import math
from ignite.metrics import Accuracy

import model.model as Models
from config_file.dataset1_configs import Config
from dataset.dataloader import Data_Loader
from worker.worker import Model_Train, Model_Finetune, Model_Test
from utils.utils import Logger, Timer, Loss_Accumulator


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='test', type=str, 
                    help='worker mode: train, finetune, test')
parser.add_argument('--model', default='momentum', type=str,
                    help='model: momentum, base, contrastive')
parser.add_argument('--signal-length', default=1024, type=int, 
                    help='signal length')
parser.add_argument('--dataset', default='dataset1', type=str,
                    help='source datasets: dataset1, dataset2')
parser.add_argument('--source-dataset', default='pretrain', type=str,
                    help='source datasets: pretrain')
parser.add_argument('--target-dataset', default='finetune_test', type=str,
                    help='target datasets: finetune_test')
parser.add_argument('--percent', default=1, type=float,
                    help='finetune dataset percentage')
parser.add_argument('--epochs', default=50, type=int, 
                    help='number of total epochs')
parser.add_argument('--batchsize', default=64, type=int,
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, 
                    help='learning rate')
parser.add_argument('--weight-decay', default=1e-3, type=float, 
                    help='weight decay')
parser.add_argument('--warmup-epochs', default=5, type=int, 
                    help='number of warmup epochs')
parser.add_argument('--logs-save-dir', default='./logs_exam', type=str, 
                    help='saving experiments logs')
parser.add_argument('--run', default='example', type=str, 
                    help='Experiment Description')


def main():
    args = parser.parse_args()
    configs = Config()

    '''save dir'''
    experiment_log_dir = os.path.join(args.logs_save_dir, args.run)
    if not os.path.exists(experiment_log_dir):   
        os.makedirs(experiment_log_dir)

    '''logger'''
    log_name = os.path.join(experiment_log_dir, '{}_logs_{}.log'.format(args.mode, time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())))
    logger = Logger(log_name)

    '''logging'''
    logger.debug('*' * 50)
    logger.debug('Mode: {}'.format(args.mode))
    logger.debug('Model: {}'.format(args.model))
    logger.debug('Signal length: {}'.format(args.signal_length))
    logger.debug('Dataset: {}'.format(args.dataset))
    logger.debug('Source Dataset: {}'.format(args.source_dataset))
    logger.debug('Target Dataset: {}'.format(args.target_dataset))
    logger.debug('Percent: {}'.format(args.percent))
    logger.debug('Epochs: {}'.format(args.epochs))
    logger.debug('Batchsize: {}'.format(args.batchsize))
    logger.debug('lr: {}'.format(args.lr))
    logger.debug('weight_decay: {}'.format(args.weight_decay))
    logger.debug('warmup_epochs: {}'.format(args.warmup_epochs))
    logger.debug('T: {}, m: {}'.format(configs.T, configs.m))
    logger.debug('Dual: {}, Cross-modal: {}'.format(configs.TF_nlayer, configs.T_decoder_layer))
    logger.debug('feature_length: {}'.format(configs.length))
    logger.debug('Heads: {}'.format(configs.nhead))
    logger.debug('tfai')
    logger.debug('*' * 50)

    '''GPU or CPU'''
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.debug('Device: {}'.format(device))

    '''datapath'''
    source_path = './dataset/{}/{}'.format(args.dataset, args.source_dataset)
    target_path = './dataset/{}/{}'.format(args.dataset, args.target_dataset)

    '''load model & data'''
    if args.mode == 'train':
        model = Models.train_model(configs)
        data_iter = Data_Loader(source_path, args, args.mode)
    elif args.mode == 'finetune':
        model = Models.finetune_model(configs)
        data_iter = Data_Loader(target_path, args, args.mode)
    else:
        model = Models.finetune_model(configs)
        data_iter = Data_Loader(target_path, args, args.mode)
    model.to(device)

    logger.debug('Model Loaded')
    logger.debug('Data Loaded')

    '''optimizer & scheduler'''
    optimizer = torch.optim.AdamW(model.parameters(),lr = args.lr, weight_decay= args.weight_decay)
    warm_up_with_cosine_lr = lambda epoch: ((epoch +1) / args.warmup_epochs ) if epoch < args.warmup_epochs \
    else 0.5 * ( math.cos((epoch - args.warmup_epochs) /(args.epochs - args.warmup_epochs) * math.pi) + 1) \
    if 0.5 * ( math.cos((epoch - args.warmup_epochs) /(args.epochs - args.warmup_epochs) * math.pi) + 1)>0.1 else 0.1
    scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=warm_up_with_cosine_lr)

    logger.debug('*' * 50)

    '''work'''
    main_worker(model, optimizer, scheduler, data_iter, device, logger, experiment_log_dir, args, configs)

    logger.debug('*' * 50)

    
def main_worker(model, optimizer, scheduler, data_iter, device, logger, experiment_log_dir, args, configs):
    logger.debug('Worker Started')
    
    '''timer'''
    timer = Timer()

    '''training mode'''
    if args.mode == 'train':
        logger.debug('Training Started')

        '''record loss & acc'''
        loss = Loss_Accumulator()
        l_c = Loss_Accumulator()
        l_a = Loss_Accumulator()
        acc = Accuracy(device)

        '''epochs'''
        for epoch in range(args.epochs):
            timer.start()
            train_loss, con_loss, match_loss, match_acc = Model_Train(model, optimizer, scheduler, data_iter, device, loss, l_c, l_a, acc, configs)
            timer.stop()
            logger.debug('\nTraining Epoch: {}. Loss: {}. Contrastive loss: {}. Matching loss: {}. Matching acc: {}.'.format(epoch, train_loss, con_loss, match_loss, match_acc))

        '''save model'''
        save_path = os.path.join(experiment_log_dir, 'saved_model')
        if not os.path.exists(save_path):   
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path,'train_state_dict.pt'))

        logger.debug('\nTrained model is saved')
    
    '''fientune mode'''
    if args.mode == 'finetune':
        logger.debug('Finetuning Stated')
        
        '''load pre-train parameters'''
        load_path = os.path.join(experiment_log_dir, 'saved_model', 'train_state_dict.pt')
        saved_model = torch.load(load_path)
        model_dict = model.state_dict()
        state_dict = {k:v for k, v in saved_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

        '''record loss & acc'''
        loss = Loss_Accumulator()
        acc = Accuracy(device)

        '''epochs'''
        for epoch in range(args.epochs):
            timer.start()
            finetune_loss, finetune_acc = Model_Finetune(model, optimizer, scheduler, data_iter, device, loss, acc)
            timer.stop()
            logger.debug('\nFinetuning Epoch: {}. Loss: {}. Accuracy: {}'.format(epoch, finetune_loss, finetune_acc))

        '''save model'''
        save_path = os.path.join(experiment_log_dir, 'saved_model')
        if not os.path.exists(save_path):   
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, 'finetune_{}_state_dict.pt'.format(args.percent)))

        logger.debug('\nFinetuned model is saved')
    
    '''test mode'''
    if args.mode == 'test':
        logger.debug('Testing Stated')

        '''load finetune parameters'''
        load_path = os.path.join(experiment_log_dir, 'saved_model', 'finetune_{}_state_dict.pt'.format(args.percent))
        model.load_state_dict(torch.load(load_path))

        '''record acc'''
        acc = Accuracy(device)

        '''testing'''
        timer.start()
        test_acc = Model_Test(model, data_iter, device, acc)
        timer.stop()
        logger.debug('\nTest Accuracy: {}'.format(test_acc))


    logger.debug('\nWork Time is: {} sec'.format(timer.sum()))

    logger.debug('\nWoker Finished')



if __name__ == '__main__':
    main()
