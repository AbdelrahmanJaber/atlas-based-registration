import glob
import os, losses, utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph

def main():
    test_dir = '/vol/miltank/users/fratzke/Documents/atlas-based-registration/'
    model_idx = -1
    weights = [1, 0.02]
    model_folder = 'TransMorph_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    model_dir = 'experiments/' + model_folder
    dict = utils.process_label()
    if os.path.exists('experiments/'+model_folder[:-1]+'.csv'):
        os.remove('experiments/'+model_folder[:-1]+'.csv')
    csv_writter(model_folder[:-1], 'experiments/' + model_folder[:-1])
    line = ''
    for i in range(46):
        line = line + ',' + dict[i]
    csv_writter(line, 'experiments/' + model_folder[:-1])

    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)

    best_model = torch.load('/vol/miltank/users/fratzke/Documents/atlas-based-registration/utils/TransMorph_mse_1_diffusion_0.02/2025-08-01_01-50-19dsc0.766.pth.tar')['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model((160, 192, 224), 'nearest')
    reg_model.cuda()

    test_set = datasets.AtlasDataset("/vol/miltank/users/fratzke/Documents/atlas-based-registration/data/json_nifti/val_pairs.json")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()

    writer = SummaryWriter(log_dir='/vol/miltank/users/fratzke/Documents/atlas-based-registration/utils/TransMorph_mse_1_diffusion_0.02/test/')
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            x_in = torch.cat((x,y),dim=1)
            x_def, flow = model(x_in)
            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            def_img = reg_model([x.cuda().float(), flow.cuda()])
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            line = utils.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            line = line #+','+str(np.sum(jac_det <= 0)/np.prod(tar.shape))
            csv_writter(line, 'experiments/' + model_folder[:-1])
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            # print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(tar.shape)))
            dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), 46)
            dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 46)
            # print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))

            plt.switch_backend('agg')

            input_img = comput_fig_broad(x)
            input_seg = comput_fig_broad(x_seg)

            warped_img = comput_fig_broad(def_img)
            warped_seg = comput_fig_broad(def_out)

            flowfield = comput_fig_broad(flow)

            atlas_img = comput_fig_broad(y)
            atlas_seg = comput_fig_broad(y_seg)

            writer.add_figure('flowfield', flowfield, stdy_idx)
            plt.close(flowfield)

            writer.add_figure('input img', input_img, stdy_idx)  
            plt.close(input_img)
            writer.add_figure('input seg', input_seg, stdy_idx) 
            plt.close(input_seg)

            writer.add_figure('atlas img', atlas_img, stdy_idx)  
            plt.close(atlas_img)
            writer.add_figure('atlas seg', atlas_seg, stdy_idx) 
            plt.close(atlas_seg)

            writer.add_figure('warped img', warped_img, stdy_idx)  
            plt.close(warped_img)
            writer.add_figure('warped seg', warped_seg, stdy_idx)  
            plt.close(warped_seg)

            stdy_idx += 1

            # flip moving and fixed images
            y_in = torch.cat((y, x), dim=1)
            y_def, flow = model(y_in)
            def_out = reg_model([y_seg.cuda().float(), flow.cuda()])
            tar = x.detach().cpu().numpy()[0, 0, :, :, :]


            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            line = utils.dice_val_substruct(def_out.long(), x_seg.long(), stdy_idx)
            line = line #+ ',' + str(np.sum(jac_det < 0) / np.prod(tar.shape))
            out = def_out.detach().cpu().numpy()[0, 0, :, :, :]
            # print('det < 0: {}'.format(np.sum(jac_det <= 0)/np.prod(tar.shape)))
            csv_writter(line, 'experiments/' + model_folder[:-1])
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            dsc_trans = utils.dice_val(def_out.long(), x_seg.long(), 46)
            dsc_raw = utils.dice_val(y_seg.long(), x_seg.long(), 46)
            # print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(), dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            stdy_idx += 1

        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))
        writer.close()

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

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