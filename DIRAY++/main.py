import argparse
import ast
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import spatial_transforms
from NCEAverage import NCEAverage
from NCECriterion import NCECriterion

from model import generate_model
from models import resnet, shufflenet, shufflenetv2, mobilenet, mobilenetv2
#
from test import get_normal_vector, split_acc_diff_threshold, cal_score ,split_acc_diff_threshold_singleattack
from utils import adjust_learning_rate, AverageMeter, Logger, get_fusion_label, l2_normalize, post_process, evaluate, \
    get_score


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--feature_dim', default=128, type=int, help='To which dimension will be embedded')
    parser.add_argument('--sample_duration', default=16, type=int, )
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--model_type', default='resnet', type=str, help='so far only resnet')
    parser.add_argument('--model_depth', default=18, type=int, help='Depth of resnet (18 | 50 | 101)')
    parser.add_argument('--shortcut_type', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--pre_train_model', default=True, type=ast.literal_eval, help='Whether use pre-trained model')
    parser.add_argument('--use_cuda', default=True, type=ast.literal_eval, help='If true, cuda is used.')
    parser.add_argument('--n_train_batch_size', default=3, type=int, help='Batch Size for normal training data')
    parser.add_argument('--a_train_batch_size', default=25, type=int, help='Batch Size for anormal training data')
    parser.add_argument('--val_batch_size', default=25, type=int, help='Batch Size for validation data')
    parser.add_argument('--learning_rate', default=0.01, type=float,
                        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.0, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight Decay')
    parser.add_argument('--epochs', default=250, type=int, help='Number of total epochs to run')
    parser.add_argument('--n_threads', default=8, type=int, help='num of workers loading dataset')
    parser.add_argument('--tracking', default=True, type=ast.literal_eval,
                        help='If true, BN uses tracking running stats')
    parser.add_argument('--norm_value', default=255, type=int,
                        help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--cal_vec_batch_size', default=20, type=int,
                        help='batch size for calculating normal driving average vector.')
    parser.add_argument('--tau', default=0.1, type=float,
                        help='a temperature parameter that controls the concentration level of the distribution of embedded vectors')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--memory_bank_size', default=200, type=int, help='Memory bank size')
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--lr_decay', default=100, type=int,
                        help='Number of epochs after which learning rate will be reduced to 1/10 of original value')
    parser.add_argument('--resume_path', default='', type=str, help='path of previously trained model')
    parser.add_argument('--resume_head_path', default='', type=str, help='path of previously trained model head')
    parser.add_argument('--initial_scales', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument('--scale_step', default=0.9, type=float, help='Scale step for multiscale cropping')
    parser.add_argument('--n_scales', default=3, type=int, help='Number of scales for multiscale cropping')
    parser.add_argument('--train_crop', default='corner', type=str,
                        help='Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)')
    parser.add_argument('--checkpoint_folder', default='./checkpoints/', type=str, help='folder to store checkpoints')
    parser.add_argument('--log_folder', default='./logs/', type=str, help='folder to store log files')
    parser.add_argument('--log_resume', default=False, type=ast.literal_eval,
                        help='True|False: a flag controlling whether to create a new log file')
    # parser.add_argument('--normvec_folder', default='./normvec/', type=str, help='folder to store norm vectors')
    # parser.add_argument('--score_folder', default='./score/', type=str, help='folder to store scores')
    parser.add_argument('--Z_momentum', default=0.9, help='momentum for normalization constant Z updates')
    parser.add_argument('--groups', default=3, type=int, help='hyper-parameters when using shufflenet')
    parser.add_argument('--width_mult', default=2.0, type=float,
                        help='hyper-parameters when using shufflenet|mobilenet')
    parser.add_argument('--val_step', default=10, type=int, help='validate per val_step epochs')
    parser.add_argument('--downsample', default=2, type=int, help='Downsampling. Select 1 frame out of N')
    parser.add_argument('--save_step', default=10, type=int, help='checkpoint will be saved every save_step epochs')


    parser.add_argument('--dataset', default='mnist', type=str,
                        help='If true, BN uses tracking running stats')
    parser.add_argument('--method', default='all', type=str,
                        help='If true, BN uses tracking running stats')

    args = parser.parse_args()
    return args


def train(model, model_head, nce_average, criterion, optimizer, epoch, args,
          batch_logger, epoch_logger, memory_bank=None):
    losses = AverageMeter()
    prob_meter = AverageMeter()
    # data_un = np.load("./lstm_encode_out/mnist_lstm_encode_train_all.npy", allow_pickle=True)#mnist
    # train_data_gen
    legth=10
    data_bn = np.load("../lstm_encode_out/mnist_lstm_encode_benign_all.npy",
                      allow_pickle=True)  # cifar train:normal_150 malicious:120
    data_ml = np.load("../lstm_encode_out/mnist_lstm_encode_malicious_all.npy", allow_pickle=True)
    data_ml=data_ml[:,:legth,:]
    data_bn=data_bn[:,:legth,:]
    model.train()
    model_head.train()

    for j in range(5):
        normal_my = torch.from_numpy(data_bn[50 * j:50 * (j + 1)])  # :50
        # 所有恶意数据一起训练
        # anormal_my = torch.from_numpy(data_un[30 * (j+1):30 * (j + 2)])
        # 选一种样本训练
        anormal_my = torch.from_numpy(data_ml[40 * j:  30 + 40 * j])  # uncertain30、kcenter60、ADFL90、ADFLK120

        data = torch.cat((normal_my, anormal_my), dim=0)
        data = torch.reshape(torch.unsqueeze(data, dim=1), (data.size()[0], 1, legth, 200, -1))

        if args.use_cuda:
            data = data.cuda()
            idx_a = torch.rand(len(anormal_my)).cuda()
            idx_n = torch.rand(len(normal_my)).cuda()
            normal_data = normal_my.cuda()

        # ================forward====================
        unnormed_vec, normed_vec = model(data)
        vec = model_head(unnormed_vec)
        n_vec = vec[0:len(normal_my)]
        a_vec = vec[len(normal_my):]
        outs, probs = nce_average(n_vec, a_vec, idx_n, idx_a, normed_vec[0:len(normal_my)])
        loss = criterion(outs)

        # ================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===========update memory bank===============
        model.eval()
        _, n = model(torch.reshape(torch.unsqueeze(normal_data, dim=1), (normal_data.size()[0], 1, legth, 200, -1)))
        n = n.detach()
        average = torch.mean(n, dim=0, keepdim=True)
        if len(memory_bank) < args.memory_bank_size:
            memory_bank.append(average)
        else:
            memory_bank.pop(0)
            memory_bank.append(average)
        model.train()

        # ===============update meters ===============
        losses.update(loss.item(), outs.size(0))
        prob_meter.update(probs.item(), outs.size(0))

        # =================logging=====================
        batch_logger.log({
            'epoch': epoch,
            'batch': j,
            'loss': losses.val,
            'probs': prob_meter.val,
            'lr': optimizer.param_groups[0]['lr']
        })
        print(
            f'Training Process is running: {epoch}/{args.epochs}  | Batch: {j} | Loss: {losses.val} ({losses.avg}) | Probs: {prob_meter.val} ({prob_meter.avg})')

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'probs': prob_meter.avg,
        'lr': optimizer.param_groups[0]['lr']
    })
    return memory_bank, losses.avg


def train_single_attack( model, model_head, nce_average, criterion, optimizer,
                        epoch, args,
                        batch_logger, epoch_logger, memory_bank=None):
    losses = AverageMeter()
    prob_meter = AverageMeter()
    # data_un = np.load("./lstm_encode_out/mnist_lstm_encode_train_all.npy", allow_pickle=True)#mnist
    # train_data_gen
    data_bn = np.load("../lstm_encode_out/mnist_lstm_encode_benign_all.npy",
                      allow_pickle=True)  # cifar train:normal_150 malicious:120
    data_ml = np.load("../lstm_encode_out/mnist_lstm_encode_malicious_all.npy", allow_pickle=True)
    model.train()
    model_head.train()

    for j in range(3):
        normal_my = torch.from_numpy(data_bn[20 * j:20 * (j + 1)])  #
        # 所有恶意数据一起训练
        # anormal_my = torch.from_numpy(data_un[30 * (j+1):30 * (j + 2)])
        # 选一种样本训练
        anormal_my = torch.from_numpy(data_ml[0+10 * j:  0+ 10 * (j + 1)])  # kcneter0、uncertian40、ADFL80、ADFLK120,knockoff160

        data = torch.cat((normal_my, anormal_my), dim=0)
        data = torch.reshape(torch.unsqueeze(data, dim=1), (data.size()[0], 1, 10, 200, -1))

        if args.use_cuda:
            data = data.cuda()
            idx_a = torch.rand(len(anormal_my)).cuda()
            idx_n = torch.rand(len(normal_my)).cuda()
            normal_data = normal_my.cuda()

        # ================forward====================
        unnormed_vec, normed_vec = model(data)
        vec = model_head(unnormed_vec)
        n_vec = vec[0:len(normal_my)]
        a_vec = vec[len(normal_my):]
        outs, probs = nce_average(n_vec, a_vec, idx_n, idx_a, normed_vec[0:len(normal_my)])
        loss = criterion(outs)

        # ================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===========update memory bank===============
        model.eval()
        _, n = model(torch.reshape(torch.unsqueeze(normal_data, dim=1), (normal_data.size()[0], 1, 10, 200, -1)))
        n = n.detach()
        average = torch.mean(n, dim=0, keepdim=True)
        if len(memory_bank) < args.memory_bank_size:
            memory_bank.append(average)
        else:
            memory_bank.pop(0)
            memory_bank.append(average)
        model.train()

        # ===============update meters ===============
        losses.update(loss.item(), outs.size(0))
        prob_meter.update(probs.item(), outs.size(0))

        # =================logging=====================
        batch_logger.log({
            'epoch': epoch,
            'batch': j,
            'loss': losses.val,
            'probs': prob_meter.val,
            'lr': optimizer.param_groups[0]['lr']
        })
        print(
            f'Training Process is running: {epoch}/{args.epochs}  | Batch: {j} | Loss: {losses.val} ({losses.avg}) | Probs: {prob_meter.val} ({prob_meter.avg})')

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'probs': prob_meter.avg,
        'lr': optimizer.param_groups[0]['lr']
    })
    return memory_bank, losses.avg


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)
    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)
    # if not os.path.exists(args.normvec_folder):
    #     os.makedirs(args.normvec_folder)
    # if not os.path.exists(args.score_folder):
    #     os.makedirs(args.score_folder)
    torch.manual_seed(args.manual_seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.manual_seed)
    if args.nesterov:
        dampening = 0
    else:
        dampening = args.dampening
    args.scales = [args.initial_scales]
    for i in range(1, args.n_scales):
        args.scales.append(args.scales[-1] * args.scale_step)
    assert args.train_crop in ['random', 'corner', 'center']
    if args.train_crop == 'random':
        crop_method = spatial_transforms.MultiScaleRandomCrop(args.scales, args.sample_size)
    elif args.train_crop == 'corner':
        crop_method = spatial_transforms.MultiScaleCornerCrop(args.scales, args.sample_size)
    elif args.train_crop == 'center':
        crop_method = spatial_transforms.MultiScaleCornerCrop(args.scales, args.sample_size, crop_positions=['c'])







    print("========================================Loading Validation Data========================================")
    val_spatial_transform = spatial_transforms.Compose([
        spatial_transforms.Scale(args.sample_size),
        spatial_transforms.CenterCrop(args.sample_size),
        spatial_transforms.ToTensor(args.norm_value),
        spatial_transforms.Normalize([0], [1])
    ])



    print(
        "============================================Generating Model============================================")

    if args.model_type == 'resnet':
        model_head = resnet.ProjectionHead(args.feature_dim, args.model_depth,args.dataset)
    elif args.model_type == 'shufflenet':
        model_head = shufflenet.ProjectionHead(args.feature_dim)
    elif args.model_type == 'shufflenetv2':
        model_head = shufflenetv2.ProjectionHead(args.feature_dim)
    elif args.model_type == 'mobilenet':
        model_head = mobilenet.ProjectionHead(args.feature_dim)
    elif args.model_type == 'mobilenetv2':
        model_head = mobilenetv2.ProjectionHead(args.feature_dim)
    if args.use_cuda:
        model_head.cuda()

    if args.method == "all":
        # len_neg, len_pos=49600,15800#150,200
        len_neg = 150
        len_pos = 250
        num_val_data = 250
    else:
        len_neg, len_pos = 30,60

    if args.resume_path == '':
        # ===============generate new model or pre-trained model===============
        model = generate_model(args)
        optimizer = torch.optim.SGD(list(model.parameters()) + list(model_head.parameters()), lr=args.learning_rate,
                                    momentum=args.momentum,
                                    dampening=dampening, weight_decay=args.weight_decay, nesterov=args.nesterov)
        nce_average = NCEAverage(args.feature_dim, len_neg, len_pos, args.tau, args.Z_momentum)
        criterion = NCECriterion(len_neg)
        begin_epoch = 1
        best_acc = 0
        memory_bank = []
    else:
        # ===============load previously trained model ===============
        args.pre_train_model = False
        model = generate_model(args)
        resume_path = os.path.join(args.checkpoint_folder, args.resume_path)
        resume_checkpoint = torch.load(resume_path)
        model.load_state_dict(resume_checkpoint['state_dict'])
        resume_head_checkpoint = torch.load(os.path.join(args.checkpoint_folder, args.resume_head_path))
        model_head.load_state_dict(resume_head_checkpoint['state_dict'])
        if args.use_cuda:
            model_head.cuda()
        optimizer = torch.optim.SGD(list(model.parameters()) + list(model_head.parameters()), lr=args.learning_rate,
                                    momentum=args.momentum,
                                    dampening=dampening, weight_decay=args.weight_decay, nesterov=args.nesterov)
        optimizer.load_state_dict(resume_checkpoint['optimizer'])
        nce_average = resume_checkpoint['nce_average']
        criterion = NCECriterion(len_neg)
        begin_epoch = resume_checkpoint['epoch'] + 1
        best_acc = resume_checkpoint['acc']
        memory_bank = resume_checkpoint['memory_bank']
        del resume_checkpoint
        torch.cuda.empty_cache()
        adjust_learning_rate(optimizer, args.learning_rate)

    print(
        "==========================================!!!START TRAINING!!!==========================================")
    cudnn.benchmark = True
    batch_logger = Logger(os.path.join(args.log_folder, 'batch.log'), ['epoch', 'batch', 'loss', 'probs', 'lr'],
                          args.log_resume)
    epoch_logger = Logger(os.path.join(args.log_folder, 'epoch.log'), ['epoch', 'loss', 'probs', 'lr'],
                          args.log_resume)
    val_logger = Logger(os.path.join(args.log_folder, 'val.log'),
                        ['epoch', 'accuracy', 'normal_acc', 'anormal_acc', 'threshold', 'acc_list',
                         'normal_acc_list', 'anormal_acc_list'], args.log_resume)

    for epoch in range(begin_epoch, begin_epoch + args.epochs + 1):
        if args.method=='all':
            memory_bank, loss = train( model, model_head, nce_average,
                                      criterion, optimizer, epoch, args, batch_logger, epoch_logger, memory_bank)
        else:
            memory_bank, loss = train_single_attack(model, model_head, nce_average,
                                      criterion, optimizer, epoch, args, batch_logger, epoch_logger, memory_bank)

        if epoch % args.val_step == 0:

            print(
                "==========================================!!!Evaluating!!!==========================================")
            normal_vec = torch.mean(torch.cat(memory_bank, dim=0), dim=0, keepdim=True)
            normal_vec = l2_normalize(normal_vec)

            model.eval()
            if args.method == 'all':
                accuracy, best_threshold, acc_n, acc_a, acc_list, acc_n_list, acc_a_list = split_acc_diff_threshold(
                    model, normal_vec, args.use_cuda)
            else:
                accuracy, best_threshold, acc_n, acc_a, acc_list, acc_n_list, acc_a_list = split_acc_diff_threshold_singleattack(
                    model, normal_vec, args.use_cuda)

            print(
                f'Epoch: {epoch}/{args.epochs} | Accuracy: {accuracy} | Normal Acc: {acc_n} | Anormal Acc: {acc_a} | Threshold: {best_threshold}')
            print(
                "==========================================!!!Logging!!!==========================================")
            val_logger.log({
                'epoch': epoch,
                'accuracy': accuracy * 100,
                'normal_acc': acc_n * 100,
                'anormal_acc': acc_a * 100,
                'threshold': best_threshold,
                'acc_list': acc_list,
                'normal_acc_list': acc_n_list,
                'anormal_acc_list': acc_a_list
            })
            if accuracy > best_acc:
                best_acc = accuracy
                print(
                    "==========================================!!!Saving!!!==========================================")
                checkpoint_path = os.path.join(args.checkpoint_folder,
                                               f'best_model_{args.model_type}.pth')
                states = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'acc': accuracy,
                    'threshold': best_threshold,
                    'nce_average': nce_average,
                    'memory_bank': memory_bank
                }
                torch.save(states, checkpoint_path)

                head_checkpoint_path = os.path.join(args.checkpoint_folder,
                                                    f'best_model_{args.model_type}_head.pth')
                states_head = {
                    'state_dict': model_head.state_dict()
                }
                torch.save(states_head, head_checkpoint_path)

        if epoch % args.save_step == 0:
            print(
                "==========================================!!!Saving!!!==========================================")
            checkpoint_path = os.path.join(args.checkpoint_folder,
                                           f'{args.model_type}_{epoch}.pth')
            states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'acc': accuracy,
                'nce_average': nce_average,
                'memory_bank': memory_bank
            }
            torch.save(states, checkpoint_path)

            head_checkpoint_path = os.path.join(args.checkpoint_folder,
                                                f'{args.model_type}_{epoch}_head.pth')
            states_head = {
                'state_dict': model_head.state_dict()
            }
            torch.save(states_head, head_checkpoint_path)

        if epoch % args.lr_decay == 0:
            lr = args.learning_rate * (0.1 ** (epoch // args.lr_decay))
            adjust_learning_rate(optimizer, lr)
            print(f'New learning rate: {lr}')
