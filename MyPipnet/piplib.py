import cv2, os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import logging
from scipy.integrate import simps

def setup_snapshot(data_name):
    # Create Snapshot Folder
    if not os.path.exists(os.path.join('./snapshots', data_name)):
        os.mkdir(os.path.join('./snapshots', data_name))
    save_dir = os.path.join('./snapshots', data_name, 'Pipnet_r18')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    return save_dir

def setup_logs(data_name):
    if not os.path.exists(os.path.join('./logs', data_name)):
        os.mkdir(os.path.join('./logs', data_name))
    log_dir = os.path.join('./logs', data_name, 'Pipnet_r18')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO)

    return log_dir


def points_flip(data_name):
    if data_name == 'data_300W':
        points_flip = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 27, 26, 25, 24, 23, 22, 21, 20,
                       19, 18, 28, 29, 30, 31, 36, 35, 34, 33, 32, 46, 45, 44, 43, 48, 47, 40, 39, 38, 37, 42, 41,
                       55, 54, 53, 52, 51, 50, 49, 60, 59, 58, 57, 56, 65, 64, 63, 62, 61, 68, 67, 66]
        points_flip = (np.array(points_flip) - 1).tolist()
        assert len(points_flip) == 68
    elif data_name == 'WFLW':
        points_flip = [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
                       9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 46, 45, 44, 43, 42, 50, 49, 48, 47, 37, 36, 35, 34, 33, 41, 40,
                       39, 38, 51, 52, 53, 54, 59, 58, 57, 56, 55, 72, 71, 70, 69, 68, 75, 74, 73, 64, 63, 62, 61,
                       60, 67, 66, 65, 82, 81, 80, 79, 78, 77, 76, 87, 86, 85, 84, 83, 92, 91, 90, 89, 88, 95, 94,
                       93, 97, 96]
        assert len(points_flip) == 98
    elif data_name == 'COFW':
        points_flip = [2, 1, 4, 3, 7, 8, 5, 6, 10, 9, 12, 11, 15, 16, 13, 14, 18, 17, 20, 19, 21, 22, 24, 23, 25,
                       26, 27, 28, 29]
        points_flip = (np.array(points_flip) - 1).tolist()
        assert len(points_flip) == 29
    elif data_name == 'AFLW':
        points_flip = [6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7, 15, 14, 13, 18, 17, 16, 19]
        points_flip = (np.array(points_flip) - 1).tolist()
        assert len(points_flip) == 19
    elif data_name == 'LaPa':
        points_flip = [33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11,
                       10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 47, 46, 45, 44, 43, 51, 50, 49, 48, 38, 37, 36, 35, 34, 42,
                       41, 40, 39, 52, 53, 54, 55, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 80, 79, 78, 77, 76,
                       83, 82, 81, 84, 71, 70, 69, 68, 67, 74, 73, 72, 75, 91, 90, 89, 88, 87, 86, 85, 96, 95, 94,
                       93, 92, 101, 100, 99, 98, 97, 104, 103, 102, 106, 105]
        points_flip = (np.array(points_flip) - 1).tolist()
        assert len(points_flip) == 106
    else:
        print('No such data!')
        exit(0)

    return points_flip

def get_label(data_name, label_file, task_type=None):
    label_path = os.path.join('.', data_name, label_file)
    with open(label_path, 'r') as f:
        labels = f.readlines()
    labels = [x.strip().split() for x in labels]
    if len(labels[0]) == 1:
        return labels

    labels_new = []
    for label in labels:
        image_name = label[0]
        target = label[1:]
        target = np.array([float(x) for x in target])
        if task_type is None:
            labels_new.append([image_name, target])
        else:
            labels_new.append([image_name, task_type, target])
    return labels_new


def get_meanface(meanface_file, num_nb):
    with open(meanface_file) as f:
        meanface = f.readlines()[0]

    meanface = meanface.strip().split()
    meanface = [float(x) for x in meanface]
    meanface = np.array(meanface).reshape(-1, 2)
    # each landmark predicts num_nb neighbors
    meanface_indices = []
    for i in range(meanface.shape[0]):
        pt = meanface[i, :]
        dists = np.sum(np.power(pt - meanface, 2), axis=1)
        indices = np.argsort(dists)
        meanface_indices.append(indices[1:1 + num_nb])

    # each landmark predicted by X neighbors, X varies
    meanface_indices_reversed = {}
    for i in range(meanface.shape[0]):
        meanface_indices_reversed[i] = [[], []]
    for i in range(meanface.shape[0]):
        for j in range(num_nb):
            meanface_indices_reversed[meanface_indices[i][j]][0].append(i)
            meanface_indices_reversed[meanface_indices[i][j]][1].append(j)

    max_len = 0
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        if tmp_len > max_len:
            max_len = tmp_len

    # tricks, make them have equal length for efficient computation
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        meanface_indices_reversed[i][0] += meanface_indices_reversed[i][0] * 10
        meanface_indices_reversed[i][1] += meanface_indices_reversed[i][1] * 10
        meanface_indices_reversed[i][0] = meanface_indices_reversed[i][0][:max_len]
        meanface_indices_reversed[i][1] = meanface_indices_reversed[i][1][:max_len]

    # make the indices 1-dim
    reverse_index1 = []
    reverse_index2 = []
    for i in range(meanface.shape[0]):
        reverse_index1 += meanface_indices_reversed[i][0]
        reverse_index2 += meanface_indices_reversed[i][1]
    return meanface_indices, reverse_index1, reverse_index2, max_len

def compute_loss_pip(outputs_map, outputs_local_x, outputs_local_y, outputs_nb_x, outputs_nb_y, labels_map, labels_local_x, labels_local_y, labels_nb_x, labels_nb_y,  loss_func, reg_func, num_nb):

    tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_map.size()
    labels_map = labels_map.view(tmp_batch*tmp_channel, -1)
    labels_max_ids = torch.argmax(labels_map, 1)
    labels_max_ids = labels_max_ids.view(-1, 1)
    labels_max_ids_nb = labels_max_ids.repeat(1, num_nb).view(-1, 1)

    outputs_local_x = outputs_local_x.view(tmp_batch*tmp_channel, -1)
    outputs_local_x_select = torch.gather(outputs_local_x, 1, labels_max_ids)
    outputs_local_y = outputs_local_y.view(tmp_batch*tmp_channel, -1)
    outputs_local_y_select = torch.gather(outputs_local_y, 1, labels_max_ids)
    outputs_nb_x = outputs_nb_x.view(tmp_batch*num_nb*tmp_channel, -1)
    outputs_nb_x_select = torch.gather(outputs_nb_x, 1, labels_max_ids_nb)
    outputs_nb_y = outputs_nb_y.view(tmp_batch*num_nb*tmp_channel, -1)
    outputs_nb_y_select = torch.gather(outputs_nb_y, 1, labels_max_ids_nb)

    labels_local_x = labels_local_x.view(tmp_batch*tmp_channel, -1)
    labels_local_x_select = torch.gather(labels_local_x, 1, labels_max_ids)
    labels_local_y = labels_local_y.view(tmp_batch*tmp_channel, -1)
    labels_local_y_select = torch.gather(labels_local_y, 1, labels_max_ids)
    labels_nb_x = labels_nb_x.view(tmp_batch*num_nb*tmp_channel, -1)
    labels_nb_x_select = torch.gather(labels_nb_x, 1, labels_max_ids_nb)
    labels_nb_y = labels_nb_y.view(tmp_batch*num_nb*tmp_channel, -1)
    labels_nb_y_select = torch.gather(labels_nb_y, 1, labels_max_ids_nb)
    labels_map = labels_map.view(tmp_batch, tmp_channel, tmp_height, tmp_width)
    loss_map = loss_func(outputs_map, labels_map)
    loss_x = reg_func(outputs_local_x_select, labels_local_x_select)
    loss_y = reg_func(outputs_local_y_select, labels_local_y_select)
    loss_nb_x = reg_func(outputs_nb_x_select, labels_nb_x_select)
    loss_nb_y = reg_func(outputs_nb_y_select, labels_nb_y_select)
    return loss_map, loss_x, loss_y, loss_nb_x, loss_nb_y

def train_model(net, train_loader, loss_func, reg_func, cls_loss_weight, reg_loss_weight, num_nb, optimizer, num_epochs, scheduler, save_dir, save_interval, device):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        logging.info('-' * 10)
        net.train()
        epoch_loss = 0.0

        for i, data in enumerate(train_loader):

            inputs, labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y = data
            inputs = inputs.to(device)
            labels_map = labels_map.to(device)
            labels_x = labels_x.to(device)
            labels_y = labels_y.to(device)
            labels_nb_x = labels_nb_x.to(device)
            labels_nb_y = labels_nb_y.to(device)
            outputs_map, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = net(inputs)
            loss_map, loss_x, loss_y, loss_nb_x, loss_nb_y = compute_loss_pip(outputs_map, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y, labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y, loss_func, reg_func, num_nb)
            loss = cls_loss_weight*loss_map + reg_loss_weight*loss_x + reg_loss_weight*loss_y + reg_loss_weight*loss_nb_x + reg_loss_weight*loss_nb_y


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%10 == 0:
                print('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> <map loss: {:.6f}> <x loss: {:.6f}> <y loss: {:.6f}> <nbx loss: {:.6f}> <nby loss: {:.6f}>'.format(
                    epoch, num_epochs-1, i, len(train_loader)-1, loss.item(), cls_loss_weight*loss_map.item(), reg_loss_weight*loss_x.item(), reg_loss_weight*loss_y.item(), reg_loss_weight*loss_nb_x.item(), reg_loss_weight*loss_nb_y.item()))
                logging.info('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> <map loss: {:.6f}> <x loss: {:.6f}> <y loss: {:.6f}> <nbx loss: {:.6f}> <nby loss: {:.6f}>'.format(
                    epoch, num_epochs-1, i, len(train_loader)-1, loss.item(), cls_loss_weight*loss_map.item(), reg_loss_weight*loss_x.item(), reg_loss_weight*loss_y.item(), reg_loss_weight*loss_nb_x.item(), reg_loss_weight*loss_nb_y.item()))

            epoch_loss += loss.item()


        epoch_loss /= len(train_loader)
        if epoch%(save_interval-1) == 0 and epoch > 0:
            filename = os.path.join(save_dir, 'epoch%d.pth' % epoch)
            torch.save(net.state_dict(), filename)
            print(filename, 'saved')
        scheduler.step()
    return net

def forward_pip(net, inputs, input_size, net_stride, num_nb):
    net.eval()
    with torch.no_grad():
        outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = net(inputs)
        tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.size()
        assert tmp_batch == 1

        outputs_cls = outputs_cls.view(tmp_batch*tmp_channel, -1)
        max_ids = torch.argmax(outputs_cls, 1)
        max_cls = torch.max(outputs_cls, 1)[0]
        max_ids = max_ids.view(-1, 1)
        max_ids_nb = max_ids.repeat(1, num_nb).view(-1, 1)

        outputs_x = outputs_x.view(tmp_batch*tmp_channel, -1)
        outputs_x_select = torch.gather(outputs_x, 1, max_ids)
        outputs_x_select = outputs_x_select.squeeze(1)
        outputs_y = outputs_y.view(tmp_batch*tmp_channel, -1)
        outputs_y_select = torch.gather(outputs_y, 1, max_ids)
        outputs_y_select = outputs_y_select.squeeze(1)

        outputs_nb_x = outputs_nb_x.view(tmp_batch*num_nb*tmp_channel, -1)
        outputs_nb_x_select = torch.gather(outputs_nb_x, 1, max_ids_nb)
        outputs_nb_x_select = outputs_nb_x_select.squeeze(1).view(-1, num_nb)
        outputs_nb_y = outputs_nb_y.view(tmp_batch*num_nb*tmp_channel, -1)
        outputs_nb_y_select = torch.gather(outputs_nb_y, 1, max_ids_nb)
        outputs_nb_y_select = outputs_nb_y_select.squeeze(1).view(-1, num_nb)

        tmp_x = (max_ids%tmp_width).view(-1,1).float()+outputs_x_select.view(-1,1)
        tmp_y = (max_ids//tmp_width).view(-1,1).float()+outputs_y_select.view(-1,1)
        tmp_x /= 1.0 * input_size / net_stride
        tmp_y /= 1.0 * input_size / net_stride

        tmp_nb_x = (max_ids%tmp_width).view(-1,1).float()+outputs_nb_x_select
        tmp_nb_y = (max_ids//tmp_width).view(-1,1).float()+outputs_nb_y_select
        tmp_nb_x = tmp_nb_x.view(-1, num_nb)
        tmp_nb_y = tmp_nb_y.view(-1, num_nb)
        tmp_nb_x /= 1.0 * input_size / net_stride
        tmp_nb_y /= 1.0 * input_size / net_stride

    return tmp_x, tmp_y, tmp_nb_x, tmp_nb_y, outputs_cls, max_cls

def compute_nme(lms_pred, lms_gt, norm):
    lms_pred = lms_pred.reshape((-1, 2))
    lms_gt = lms_gt.reshape((-1, 2))
    nme = np.mean(np.linalg.norm(lms_pred - lms_gt, axis=1)) / norm
    return nme

def compute_fr_and_auc(nmes, thres=0.1, step=0.0001):
    num_data = len(nmes)
    xs = np.arange(0, thres + step, step)
    ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
    fr = 1.0 - ys[-1]
    auc = simps(ys, x=xs) / thres
    return fr, auc
