data_name = 'data_300W'
net_stride = 32
batch_size = 16
init_lr = 0.0001
num_epochs = 60
decay_steps = [30, 50]
input_size = 256
cls_loss_weight = 10
reg_loss_weight = 1
num_lms = 68
save_interval = num_epochs
num_nb = 10
use_gpu = True
gpu_id = 2