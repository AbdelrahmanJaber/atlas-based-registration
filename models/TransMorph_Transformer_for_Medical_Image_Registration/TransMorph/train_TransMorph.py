from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
import torch.nn.functional as F
import json
import random
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import make_grid
import pathlib
from datetime import datetime


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    input_json = "/vol/miltank/projects/practical_sose25/atlas_based_registeration/utils/preprocessed_ct_seg_patella_pairs.json"
    batch_size = 1
    weights = [1, 0.02] # loss weights
    save_dir = 'TransMorph_mse_{}_diffusion_{}/{}'.format(weights[0], weights[1], timestamp)
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    sys.stdout = Logger('logs/'+save_dir)
    lr = 0.0001 # learning rate
    epoch_start = 0
    max_epoch = 100 #500 #max traning epoch
    cont_training = False #if continue training

    '''
    Initialize model
    '''
    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 394
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-2])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-2]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    # train_composed = transforms.Compose([trans.RandomFlip(0),
    #                                      trans.NumpyType((np.float32, np.float32)),
    #                                      ])

    # val_composed = transforms.Compose([trans.Seg_norm(), #rearrange segmentation label to 1 to 46
    #                                    trans.NumpyType((np.float32, np.int16)),
    #                                     ])

    # # --- Parameter ---
    train_output = "/vol/miltank/projects/practical_sose25/atlas_based_registeration/data/json_nifti/train_pairs.json"
    val_output = "/vol/miltank/projects/practical_sose25/atlas_based_registeration/data/json_nifti/val_pairs.json"

    # TODO uncomment to split datasets
    # val_ratio = 0.2  # 20% valid

    # with open(input_json, "r") as f:
    #     data = json.load(f)

    # keys = list(data.keys())
    # # random.seed(random_seed)
    # # random.shuffle(keys)

    # split_index = int(len(keys) * (1 - val_ratio))
    # train_keys = keys[:split_index]
    # val_keys = keys[split_index:]

    # train_data = {k: data[k] for k in train_keys}
    # val_data = {k: data[k] for k in val_keys}

    # with open(train_output, "w") as f:
    #     json.dump(train_data, f, indent=2)

    # with open(val_output, "w") as f:
    #     json.dump(val_data, f, indent=2)

    # print(f"Train: {len(train_data)}, Valid: {len(val_data)}")

    train_set = datasets.AtlasDataset(train_output)
    val_set = datasets.AtlasDataset(val_output)

    print(f"Number of training samples: {len(train_set)}", flush=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    # criterion = losses.NCC_vxm() #TODO uncomment to use local NCC
    criterion = nn.MSELoss()
    criterions = [criterion]
    # criterions += [losses.Grad3d()] #TODO uncomment to use l1 Grad
    criterions += [losses.Grad3d(penalty='l2')]

    backup_dir = pathlib.Path(save_dir + "/" + timestamp)
    backup_dir.mkdir(parents=True, exist_ok=True)

    best_dsc = 0
    best_epoch = 0

    writer = SummaryWriter(log_dir='logs/'+save_dir)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0

        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]

            x_in = torch.cat((x,y), dim=1)
            output = model(x_in)
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del x_in
            del output
            # flip fixed and moving images
            loss = 0
            x_in = torch.cat((y, x), dim=1)
            output = model(x_in)
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], x) * weights[n]
                loss_vals[n] += curr_loss
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item()/2, loss_vals[1].item()/2))
            # print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item()/2))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        atlas_list = []
        warped_list = []
        input_list = []

        atlas_seg_list = []
        warped_seg_list = []
        input_seg_list = []

        flow_list = []
        grid_list = []
        eval_dsc = utils.AverageMeter()

        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                x_in = torch.cat((x, y), dim=1)
                grid_img = mk_grid_img(8, 1, config.img_size)
                output = model(x_in)

                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                def_img = reg_model([x.cuda().float(), output[1].cuda()])
                def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])

                flowfield = output[1]

                input_list.append(x.squeeze(0).cpu().numpy())
                input_seg_list.append(x_seg.squeeze(0).cpu().numpy())
                warped_list.append(def_img.squeeze(0).cpu().detach().numpy())
                warped_seg_list.append(def_out.squeeze(0).cpu().detach().numpy())
                flow_list.append(flowfield.squeeze(0).cpu().detach().numpy())
                atlas_list.append(y.squeeze(0).cpu().numpy())
                atlas_seg_list.append(y_seg.squeeze(0).cpu().numpy())
                grid_list.append(def_grid.squeeze(0).cpu().detach().numpy())

                dsc = utils.dice_val(def_out.round().long(), y_seg.long(), 2)
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)


        if(eval_dsc.avg > best_dsc):
            print(f"replacing {best_epoch} with {epoch}", flush=True)
            file_map_current = {
                f"epoch_{epoch}_atlas.npy": atlas_list,
                f"epoch_{epoch}_atlas_segs.npy": atlas_seg_list,
                f"epoch_{epoch}_warped.npy": warped_list,
                f"epoch_{epoch}_warped_segs.npy": warped_seg_list,
                f"epoch_{epoch}_input.npy": input_list,
                f"epoch_{epoch}_input_segs.npy": input_seg_list,
                f"epoch_{epoch}_flow.npy": flow_list,
                f"epoch_{epoch}_grid.npy": grid_list
            }

            file_map_old = {
                f"epoch_{best_epoch}_atlas.npy": atlas_list,
                f"epoch_{best_epoch}_atlas_segs.npy": atlas_seg_list,
                f"epoch_{best_epoch}_warped.npy": warped_list,
                f"epoch_{best_epoch}_warped_segs.npy": warped_seg_list,
                f"epoch_{best_epoch}_input.npy": input_list,
                f"epoch_{best_epoch}_input_segs.npy": input_seg_list,
                f"epoch_{best_epoch}_flow.npy": flow_list,
                f"epoch_{best_epoch}_grid.npy": grid_list
            }

            for filename, data in file_map_current.items():
                filepath = os.path.join(backup_dir, filename)
                print(f"save {filename} epoch {epoch}", flush=True)
                if os.path.exists(filepath):
                    os.remove(filepath)

                np.save(filepath, data)

            if(best_epoch != epoch):
                #delete old files
                for filename, data in file_map_old.items():
                    filepath = os.path.join(backup_dir, filename)
                    print(f"delete files epoch {best_epoch}", flush=True)
                    if os.path.exists(filepath):
                        os.remove(filepath)

            best_epoch = epoch


        best_dsc = max(eval_dsc.avg, best_dsc)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir=save_dir, filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))

        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
        plt.switch_backend('agg')

        input_img = comput_fig_broad(x)
        input_seg = comput_fig_broad(x_seg)

        warped_img = comput_fig_broad(def_img)
        warped_seg = comput_fig_broad(def_out)

        flowfield = comput_fig_broad(flowfield)
        grid = comput_fig_broad(def_grid)

        atlas_img = comput_fig_broad(y)
        atlas_seg = comput_fig_broad(y_seg)

        writer.add_figure('flowfield', flowfield, epoch)
        plt.close(flowfield)
        writer.add_figure('grid', grid, epoch)
        plt.close(grid)

        writer.add_figure('input img', input_img, epoch) 
        plt.close(input_img)
        writer.add_figure('input seg', input_seg, epoch) 
        plt.close(input_seg)

        writer.add_figure('atlas img', atlas_img, epoch) 
        plt.close(atlas_img)
        writer.add_figure('atlas seg', atlas_seg, epoch) 
        plt.close(atlas_seg)

        writer.add_figure('warped img', warped_img, epoch) 
        plt.close(warped_img)
        writer.add_figure('warped seg', warped_seg, epoch) 
        plt.close(warped_seg)

        loss_all.reset()

    writer.close()

def choose_slice(mask):
    # nimm Mittelpunkt des Bounding-Boxes der Maske
    zz = torch.where(mask>0)[2]
    z_mid = int((zz.min()+zz.max())/2) if len(zz)>0 else mask.shape[2]//2
    return z_mid

def mask_to_vis(mask):
    """
    mask : torch.Tensor  [B,1,D,H,W] oder [B,1,H,W]
           Werte 0 / 16 (oder 0 / 1)
    Ausgabe : [B,1,H,W]  Float32 in [0,1]
    """
    m = mask.detach().cpu()
    if m.ndim == 5:           # 3-D → Mittel-Slice
        m = m[:, :, m.shape[2] // 2]   # (B,1,H,W)

    # ► Binarisieren & auf Float bringen
    m = (m > 0).float()       # 0 → 0  ,  alles >0 → 1
    return m                  # Werte 0 oder 1

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def comput_fig_broad(img):
    # img: Tensor [1, 1, D, H, W]
    img = img.detach().cpu().numpy()[0, 0]  # [D, H, W]

    # Wähle 16 gleichmäßig verteilte Slices aus dem Volumen
    depth = img.shape[0]
    if depth < 16:
        idxs = list(range(depth))
        idxs += [depth - 1] * (16 - depth)  # auffüllen falls weniger als 16 Slices
    else:
        idxs = np.linspace(0, depth - 1, 16, dtype=int)

    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i, idx in enumerate(idxs):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[idx], cmap='gray', vmin=0, vmax=1)
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def to_img(t, hu_min=-1000, hu_max=2000):
    t = t.detach().cpu().float().numpy()
    if t.ndim == 5: t = t[0,0]             # (D,H,W)
    t = np.clip(t, hu_min, hu_max)
    t = (t - hu_min) / (hu_min - hu_max) * -1  # [0,1]
    return torch.from_numpy(t)[None,None]      # zurück wie comput_fig erwartet

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    # model_lists = natsorted(glob.glob(save_dir + '*'))
    # while len(model_lists) > max_model_num:
    #     # os.remove(model_lists[0])
    #     model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    # GPU_iden = 1
    # GPU_num = torch.cuda.device_count()
    # print('Number of GPU: ' + str(GPU_num))
    # for GPU_idx in range(GPU_num):
    #     GPU_name = torch.cuda.get_device_name(GPU_idx)
    #     print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    # torch.cuda.set_device(GPU_iden)
    # GPU_avai = torch.cuda.is_available()
    # print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    # print('If the GPU is available? ' + str(GPU_avai))

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        GPU_iden = 0
        torch.cuda.set_device(GPU_iden)
        print(f"Using GPU: {torch.cuda.get_device_name(GPU_iden)}")
    else:
        print("No GPU available. Using CPU.")
        device = torch.device("cpu")

    main()