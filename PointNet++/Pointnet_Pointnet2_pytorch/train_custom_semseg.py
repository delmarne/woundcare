import argparse
import os
import torch
import torch.nn as nn
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# Import your custom segmentation dataloader
from data_utils.Wound_Data_Loader_RHL import WoundSegDataset
from shape_prior_loss import ClinicalShapePriorLoss

classes = ['healthy_skin', 'wound_bed']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {i: cat for i, cat in enumerate(seg_classes.keys())}

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Wound Segmentation Training')
    parser.add_argument('--model', type=str, default='wound_seg_gnn', help='model name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training')
    parser.add_argument('--epoch', default=50, type=int, help='Epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default='wound_semseg_01', help='Log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay')
    
    # Data arguments
    parser.add_argument('--json_path', type=str, required=True, help='path to dict_wounds.json')
    parser.add_argument('--data_dir', type=str, required=True, help='directory with H5 files')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 2
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    print("Loading training data ...")
    TRAIN_DATASET = WoundSegDataset(json_path=args.json_path, data_dir=args.data_dir, split='train', num_points=NUM_POINT)
    print("Loading test data ...")
    TEST_DATASET = WoundSegDataset(json_path=args.json_path, data_dir=args.data_dir, split='test', num_points=NUM_POINT)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    # Define class weights to combat severe class imbalance
    # Class 0: healthy_skin (weight = 1.0)
    # Class 1: wound_bed (weight = 10.0) -> Adjust this up or down based on results
    class_weights = torch.tensor([1.0, 5.0]).float().cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
    shape_prior_criterion = ClinicalShapePriorLoss(target_aspect_ratio_max=3.0, target_depth_variance_max=0.1).cuda()
    alpha = 0.0  # Weight of the shape prior (tune this between 0.01 and 0.5)
    classifier.apply(inplace_relu)

    start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    for epoch in range(start_epoch, args.epoch):
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)  # [B, 3, N]

            # Forward pass through GNN
            seg_pred = classifier(points)  # Returns [B, 2, N]

            # --- CALCULATE LOSSES ---

            # 1. Standard Segmentation Loss (CrossEntropy)
            # Reshape for CrossEntropyLoss [B*N, 2] vs [B*N]
            seg_pred_reshaped = seg_pred.transpose(2, 1).contiguous().view(-1, NUM_CLASSES)
            target_flat = target.view(-1)
            loss_ce = criterion(seg_pred_reshaped, target_flat)

            # 2. Clinical Shape Prior Loss
            # Pass the UNRESHAPED logits [B, 2, N] and the XYZ points [B, 3, N]
            loss_prior = shape_prior_criterion(seg_pred, points[:, :3, :])

            # 3. Total Loss
            loss = loss_ce + (alpha * loss_prior)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            batch_label = target.cpu().data.numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss.item()
            
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        '''Evaluate on test set'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            log_string('---- EVALUATION ----')
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred = classifier(points)
                pred_val = seg_pred.transpose(2, 1).contiguous().cpu().data.numpy() # [B, N, 2]
                seg_pred = seg_pred.transpose(2, 1).contiguous().view(-1, NUM_CLASSES)
                
                batch_label = target.cpu().data.numpy()
                target_flat = target.view(-1)
                
                loss = criterion(seg_pred, target_flat)
                loss_sum += loss.item()
                
                pred_val = np.argmax(pred_val, 2) # [B, N]
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))
            
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class IoU: %f' % (mIoU))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                    total_correct_class[l] / float(total_iou_deno_class[l] + 1e-6))
            log_string(iou_per_class_str)

            if mIoU >= best_iou:
                best_iou = mIoU
                savepath = str(checkpoints_dir) + '/best_model.pth'
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving best model....')
            log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1

if __name__ == '__main__':
    args = parse_args()
    main(args)