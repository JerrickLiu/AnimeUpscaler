import os
import sys
import time
import copy
import shutil
import random

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import config
import utils
from loss import Loss
from data import AnimeDataset, AnimeVectorizedDataset, AnimeInterpDataset

from svg_encoder import *
from data.render_segments import *


##### Parse CmdLine Arguments #####
args, unparsed = config.get_args()
cwd = os.getcwd()
print(args)


##### TensorBoard & Device Setup #####
if args.mode != 'test':
    writer = SummaryWriter('logs/%s' % args.exp_name)

device = torch.device('cuda' if args.cuda else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

LOSS_0 = 0

def load_dataset(args):
    """ 
    Load the dataset. If the dataset is the anime dataset, use our custom dataset class.
    """

    # TODO: Make this more robust
    if args.dataset == 'anime':
        train_loader = AnimeDataset.get_loader('', args.data_root + '/metadata/all_scenes.csv', args.data_root + '/extracted_frames', args.batch_size, True, args.num_workers, args.test_mode)
        test_loader = AnimeDataset.get_loader('', args.test_data_root + '/metadata/all_scenes.csv', args.test_data_root + '/extracted_frames', args.batch_size, True, args.num_workers, args.test_mode)

    elif args.dataset == 'anime_vectorized':
        train_loader = AnimeVectorizedDataset.get_loader('', args.data_root + '/metadata/all_scenes.csv', args.data_root + '/extracted_frames', args.svg_dir + '/extracted_frames_vectorized', args.batch_size, True, args.num_workers, args.test_mode)

        test_loader = AnimeVectorizedDataset.get_loader('', args.test_data_root + '/metadata/all_scenes.csv', args.test_data_root + '/extracted_frames', args.test_svg_dir + '/extracted_frames_vectorized', args.batch_size, True, args.num_workers, args.test_mode)

    elif args.dataset == 'anime_interp':
        train_loader = AnimeInterpDataset.get_loader('', args.data_root, args.svg_dir, args.batch_size, True, args.num_workers, args.test_mode)

        test_loader = AnimeInterpDataset.get_loader('', args.test_data_root, args.test_svg_dir, args.batch_size, True, args.num_workers, args.test_mode)

    else:
        train_loader, test_loader = utils.load_dataset(
            args.dataset, args.data_root, args.batch_size, args.test_batch_size, args.num_workers, args.test_mode)
    return train_loader, test_loader


def build_model(args):
    """
    Build the model. The model consists of our SVG encoder as well as context embedder. We also have a intermediate vector predictor model, which then finally gets passed into CAIN.
    """

    if args.model.lower() == 'cain_encdec':
        from model.cain_encdec import CAIN_EncDec
        print('Building model: CAIN_EncDec')
        model = CAIN_EncDec(depth=args.depth, start_filts=32)

    elif args.model.lower() == 'cain':
        from model.cain import CAIN
        print("Building model: CAIN")
        model = CAIN(depth=args.depth, vector_intermediate=args.vector_intermediate)

        if args.vector_model.lower() == 'attention':
            from model.vector_cain import VectorCAIN
            print("Building model: VectorCAIN")
            vector_model = VectorCAIN(depth=args.depth)

        else:
            print("Building model: Naive VectorCAIN")

            # Naive vector predictor. Making CAIN smaller
            VECTOR_CAIN_N_RESGROUPS = 2
            VECTOR_CAIN_N_RESBLOCKS = 6

            vector_model = CAIN(n_resgroups=VECTOR_CAIN_N_RESGROUPS, n_resblocks=VECTOR_CAIN_N_RESBLOCKS, depth=args.depth, in_channels=3)

    elif args.model.lower() == 'cain_noca':
        from model.cain_noca import CAIN_NoCA
        print("Building model: CAIN_NoCA")
        model = CAIN_NoCA(depth=args.depth)

    else:
        raise NotImplementedError("Unknown model!")

    # Builds SVG Encoder and Context Embedder
    svg_encoder, context_embedder = build_context_embedder()

    # Just make every model to DataParallel
    model = torch.nn.DataParallel(model).to(device)
    svg_encoder = svg_encoder.to(device)
    context_embedder = context_embedder.to(device)
    if vector_model is not None:
        vector_model = torch.nn.DataParallel(vector_model).to(device)

    return model, vector_model, svg_encoder, context_embedder


def build_context_embedder():
    # Input size is 13 because we have segments of dim 8 + colors of dim 3 + transforms of dim 2
    INPUT_SIZE = 13
    EMBED_SIZE = 64
    HIDDEN_SIZE = 32
    OUTPUT_SIZE = 32
    BIDIRECTIONAL = False
    NUM_LAYERS = 2

    svg_encoder = SVGEncoder(input_size=INPUT_SIZE, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, bidirectional=BIDIRECTIONAL, num_layers=NUM_LAYERS)

    context_embedder = ContextEmbedder(latent_dim=OUTPUT_SIZE)

    return svg_encoder, context_embedder


def train(args, train_loader, model, criterion, optimizer, lpips_model, epoch):
    global LOSS_0
    losses, psnrs, ssims, lpips = utils.init_meters(args.loss)
    model.train()
    criterion.train()

    t = time.time()
    # Breaking compatability with original CAIN datasets
    for i, (images, t) in enumerate(train_loader):

        # Build input batch
        # im1, im2, gt = utils.build_input(images, imgpaths)
        images = images.cuda()
        # print(images[0].shape)
        # print(images.shape)
        im1 = images[:, 0, ...]
        im2 = images[:, 2, ...]
        gt = images[:, 1, ...]
        # print(im1.shape)

        # Forward
        optimizer.zero_grad()
        out, feats = model(im1, im2)
        loss, loss_specific = criterion(out, gt, None, feats)

        # Save loss values
        for k, v in losses.items():
            if k != 'total':
                v.update(loss_specific[k].item())
        if LOSS_0 == 0:
            LOSS_0 = loss.data.item()
        losses['total'].update(loss.item())

        # Backward (+ grad clip) - if loss explodes, skip current iteration
        loss.backward()
        if loss.data.item() > 10.0 * LOSS_0:
            print(max(p.grad.data.abs().max() for p in model.parameters()))
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        # Calc metrics & print logs
        if i % args.log_iter == 0:
            utils.eval_metrics(out, gt, psnrs, ssims, lpips, lpips_model)

            # print(time.time() - t)
            print(' Train Epoch: {} [{}/{}]   Loss: {:.6f}  PSNR: {:.4f}'.format(epoch, i, len(train_loader), losses['total'].avg, psnrs.avg))
            # print('Train Epoch: {} [{}/{}]   Loss: {:.6f}  PSNR: {:.4f}  Time({:.2f})'.format(epoch, i, len(train_loader), losses['total'].avg, psnrs.avg, time.time() - t))
            
            # Log to TensorBoard
            utils.log_tensorboard(writer, losses, psnrs.avg, ssims.avg, lpips.avg,
                optimizer.param_groups[-1]['lr'], epoch * len(train_loader) + i)

            # Reset metrics
            losses, psnrs, ssims, lpips = utils.init_meters(args.loss)
            t = time.time()


def test(args, test_loader, model, criterion, optimizer, epoch, eval_alpha=0.5):
    print('Evaluating for epoch = %d' % epoch)
    losses, psnrs, ssims, lpips = utils.init_meters(args.loss)
    model.eval()
    criterion.eval()

    save_folder = 'test%03d' % epoch
    if args.dataset == 'snufilm':
        save_folder = os.path.join(save_folder, args.dataset, args.test_mode)
    else:
        save_folder = os.path.join(save_folder, args.dataset)
    save_dir = os.path.join('checkpoint', args.exp_name, save_folder)
    utils.makedirs(save_dir)
    save_fn = os.path.join(save_dir, 'results.txt')
    if not os.path.exists(save_fn):
        with open(save_fn, 'w') as f:
            f.write('For epoch=%d\n' % epoch)

    t = time.time()
    with torch.no_grad():
        for i, (images, imgpaths) in enumerate(tqdm(test_loader)):
            images = images.cuda()
            # Build input batch
            # im1, im2, gt = utils.build_input(images, imgpaths, is_training=False)
            im1 = images[:, 0, ...]
            im2 = images[:, 2, ...]
            gt = images[:, 1, ...]

            # Forward
            out, feats = model(im1, im2)

            # Save loss values
            loss, loss_specific = criterion(out, gt, None, feats)
            for k, v in losses.items():
                if k != 'total':
                    v.update(loss_specific[k].item())
            losses['total'].update(loss.item())

            # Evaluate metrics
            utils.eval_metrics(out, gt, psnrs, ssims, lpips)

            # Log examples that have bad performance
            # if (ssims.val < 0.9 or psnrs.val < 25) and epoch > 50:
                # print(imgpaths)
                # print("\nLoss: %f, PSNR: %f, SSIM: %f, LPIPS: %f" %
                      # (losses['total'].val, psnrs.val, ssims.val, lpips.val))
                # print(imgpaths[1][-1])

            # # Save result images
            # if ((epoch + 1) % 1 == 0 and i < 20) or args.mode == 'test':
                # savepath = os.path.join('checkpoint', args.exp_name, save_folder)

                # for b in range(images[0].size(0)):
                    # paths = imgpaths[1][b].split('/')
                    # fp = os.path.join(savepath, paths[-3], paths[-2])
                    # if not os.path.exists(fp):
                        # os.makedirs(fp)
                    # # remove '.png' extension
                    # fp = os.path.join(fp, paths[-1][:-4])
            #         utils.save_image(out[b], "%s.png" % fp)
                    
    # Print progress
    print('im_processed: {:d}/{:d} {:.3f}s   \r'.format(i + 1, len(test_loader), time.time() - t))
    print("Loss: %f, PSNR: %f, SSIM: %f, LPIPS: %f\n" %
          (losses['total'].avg, psnrs.avg, ssims.avg, lpips.avg))

    # Save psnr & ssim
    save_fn = os.path.join('checkpoint', args.exp_name, save_folder, 'results.txt')
    with open(save_fn, 'a') as f:
        f.write("PSNR: %f, SSIM: %f, LPIPS: %f\n" %
                (psnrs.avg, ssims.avg, lpips.avg))

    # Log to TensorBoard
    if args.mode != 'test':
        utils.log_tensorboard(writer, losses, psnrs.avg, ssims.avg, lpips.avg,
            optimizer.param_groups[-1]['lr'], epoch * len(train_loader) + i, mode='test')

    return losses['total'].avg, psnrs.avg, ssims.avg, lpips.avg

# TODO: Finish train and test on SVGS
def train_vectorized(args, train_loader, model, vector_model, svg_encoder, context_embedder, criterion, optimizer, lpips_model, epoch):
    global LOSS_0
    vector_model = vector_model.cuda()
    losses, psnrs, ssims, lpips = utils.init_meters(args.loss)
    model.train()
    criterion.train()

    t = time.time()
    # Breaking compatability with original CAIN datasets
    for i, (images, svgs, num_segments, svg_files, t, svg_prepad_info) in enumerate(train_loader):
        images, svgs = images.to(device), svgs.to(device)

        optimizer.zero_grad()
        context_vectors = embed_svgs(svgs, svg_encoder, context_embedder)
        norms = torch.norm(context_vectors, dim=2, keepdim=True)
        context_normed = context_vectors / norms
        frame1_batch = context_normed[0::2, :, :]
        frame3_batch = context_normed[1::2, :, :]
        sim = torch.bmm(frame1_batch, frame3_batch.transpose(1, 2))

        masks = batch_render_clusters_correspondence(svg_files, svg_prepad_info, sim, num_segments)

        im1 = images[:, 0, ...]
        im2 = images[:, 2, ...]
        gt = images[:, 1, ...]

        vector_model_outputs = []
        mask_clones = masks.clone()
        for c in range(masks.shape[2]):
            im1_clone = im1.clone()
            im2_clone = im2.clone()
            v_output = vector_model(im1_clone * mask_clones[:, 0, c:c+1, ...], im2_clone * mask_clones[:, 1, c:c+1, ...])[0][:, :3]
            vector_model_outputs.append(v_output)
        
        stacked = torch.stack(vector_model_outputs, dim=0)
        intermediate = torch.sum(stacked, dim=0)

        # Forward for refinement
        out, feats = model(im1, im2, intermediate)
        loss, loss_specific = criterion(out, gt, intermediate, None, feats)
        loss = loss + torch.mean(sim)
        
        # Save loss values
        for k, v in losses.items():
            if k != 'total':
                v.update(loss_specific[k].item())

        if LOSS_0 == 0:
            LOSS_0 = loss.data.item()
        losses['total'].update(loss.item())

        # Backward (+ grad clip) - if loss explodes, skip current iteration
        loss.backward()
        if loss.data.item() > 10.0 * LOSS_0:
            print(max(p.grad.data.abs().max() for p in model.parameters()))
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        # Calc metrics & print logs
        if i % args.log_iter == 0:
            utils.eval_metrics(out, gt, psnrs, ssims, lpips, lpips_model)

            print(' Train Epoch: {} [{}/{}]   Loss: {:.6f}  PSNR: {:.4f}'.format(epoch, i, len(train_loader), losses['total'].avg, psnrs.avg))
            
            # Log to TensorBoard
            utils.log_tensorboard(writer, losses, psnrs.avg, ssims.avg, lpips.avg,
                optimizer.param_groups[-1]['lr'], epoch * len(train_loader) + i)
            step = epoch * len(train_loader) + i
            writer.add_images('mask1', masks[0, 0].unsqueeze(1), global_step=step)
            writer.add_images('mask3', masks[0, 1].unsqueeze(1), global_step=step)
            writer.add_image('intermed', intermediate[0], global_step=step)
            writer.add_image('out', out[0], global_step=step)
            writer.add_image('gt', gt[0], global_step=step)
            writer.add_image('input1', im1[0], global_step=step)
            writer.add_image('input2', im2[0], global_step=step)
            # plt.imshow(im2[0].detach().cpu().numpy().transpose(1, 2, 0))
            # plt.show()
            # plt.imshow(gt[0].detach().cpu().numpy().transpose(1, 2, 0))
            # plt.show()
            # Reset metrics
            losses, psnrs, ssims, lpips = utils.init_meters(args.loss)
            t = time.time()

def test_vectorized(args, test_loader, train_loader, model, vector_model, svg_encoder, context_embedder, criterion, optimizer, epoch, eval_alpha=0.5):
    print('Evaluating for epoch = %d' % epoch)
    losses, psnrs, ssims, lpips = utils.init_meters(args.loss)
    model.eval()
    criterion.eval()

    save_folder = 'test%03d' % epoch
    if args.dataset == 'snufilm':
        save_folder = os.path.join(save_folder, args.dataset, args.test_mode)
        
    else:
        save_folder = os.path.join(save_folder, args.dataset)
    save_dir = os.path.join('checkpoint', args.exp_name, save_folder)
    utils.makedirs(save_dir)
    save_fn = os.path.join(save_dir, 'results.txt')
    if not os.path.exists(save_fn):
        with open(save_fn, 'w') as f:
            f.write('For epoch=%d\n' % epoch)

    t = time.time()
    with torch.no_grad():
        for i, (images, svgs, num_segments, svg_files, time_deltas, svg_prepad_info) in enumerate(tqdm(test_loader)):
            images, svgs = images.to(device), svgs.to(device)

            optimizer.zero_grad()
            context_vectors = embed_svgs(svgs, svg_encoder, context_embedder)
            norms = torch.norm(context_vectors, dim=2, keepdim=True)
            context_normed = context_vectors / norms
            frame1_batch = context_normed[0::2, :, :]
            frame3_batch = context_normed[1::2, :, :]
            sim = torch.bmm(frame1_batch, frame3_batch.transpose(1, 2))

            masks = batch_render_clusters_correspondence(svg_files, svg_prepad_info, sim, num_segments)

            im1 = images[:, 0, ...]
            im2 = images[:, 2, ...]
            gt = images[:, 1, ...]

            vector_model_outputs = []
            mask_clones = masks.clone()
            for c in range(masks.shape[2]):
                im1_clone = im1.clone()
                im2_clone = im2.clone()
                v_output = vector_model(im1_clone * mask_clones[:, 0, c:c+1, ...], im2_clone * mask_clones[:, 1, c:c+1, ...])[0][:, :3]
                vector_model_outputs.append(v_output)
            
            stacked = torch.stack(vector_model_outputs, dim=0)
            intermediate = torch.sum(stacked, dim=0)

            # Forward for refinement
            out, feats = model(im1, im2, intermediate)
            loss, loss_specific = criterion(out, gt, intermediate, None, feats)
            loss = loss + torch.mean(sim)

            # Save loss values
            loss, loss_specific = criterion(out, gt, intermediate, None, feats)
            loss = loss + torch.mean(sim)
            for k, v in losses.items():
                if k != 'total':
                    v.update(loss_specific[k].item())
            losses['total'].update(loss.item())

            # Evaluate metrics
            utils.eval_metrics(out, gt, psnrs, ssims, lpips)

            # Log examples that have bad performance
            # if (ssims.val < 0.9 or psnrs.val < 25) and epoch > 50:
                # print(imgpaths)
                # print("\nLoss: %f, PSNR: %f, SSIM: %f, LPIPS: %f" %
                      # (losses['total'].val, psnrs.val, ssims.val, lpips.val))
                # print(imgpaths[1][-1])

            # # Save result images
            # if ((epoch + 1) % 1 == 0 and i < 20) or args.mode == 'test':
                # savepath = os.path.join('checkpoint', args.exp_name, save_folder)

                # for b in range(images[0].size(0)):
                    # paths = imgpaths[1][b].split('/')
                    # fp = os.path.join(savepath, paths[-3], paths[-2])
                    # if not os.path.exists(fp):
                        # os.makedirs(fp)
                    # # remove '.png' extension
                    # fp = os.path.join(fp, paths[-1][:-4])
            #         utils.save_image(out[b], "%s.png" % fp)
                    
    # Print progress
    print('im_processed: {:d}/{:d} {:.3f}s   \r'.format(i + 1, len(test_loader), time.time() - t))
    print("Loss: %f, PSNR: %f, SSIM: %f, LPIPS: %f\n" %
          (losses['total'].avg, psnrs.avg, ssims.avg, lpips.avg))

    # Save psnr & ssim
    save_fn = os.path.join('checkpoint', args.exp_name, save_folder, 'results.txt')
    with open(save_fn, 'a') as f:
        f.write("PSNR: %f, SSIM: %f, LPIPS: %f\n" %
                (psnrs.avg, ssims.avg, lpips.avg))

    # Log to TensorBoard
    if args.mode != 'test':
        utils.log_tensorboard(writer, losses, psnrs.avg, ssims.avg, lpips.avg,
            optimizer.param_groups[-1]['lr'], epoch * len(train_loader) + i, mode='test')

    return losses['total'].avg, psnrs.avg, ssims.avg, lpips.avg

""" Entry Point """
def main(args):

    # Get dataloaders
    train_loader, test_loader = load_dataset(args)

    # Get model
    model, vector_model, svg_encoder, context_embedder = build_model(args)

    criterion = Loss(args)

    args.radam = False
    if args.radam:
        from radam import RAdam
        optimizer = RAdam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    else:
        from torch.optim import Adam
        optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    print('# of parameters: %d' % sum(p.numel() for p in model.parameters()))

    # If resume, load checkpoint: model + optimizer
    if args.resume:
        # utils.load_checkpoint(args, model, optimizer)
        print("Loading checkpoint...")
        checkpoint = torch.load('pretrained_cain.pth')
        args.start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        del checkpoint
        print("Loaded!")

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True)


    # Initialize LPIPS model if used for evaluation
    lpips_model = utils.init_lpips_eval() if args.lpips else None

    if args.mode == 'test':
        _, _, _, _ = test_vectorized(
                    args=args,
                    test_loader=test_loader,
                    train_loader=train_loader,
                    model=model,
                    vector_model=vector_model,
                    svg_encoder=svg_encoder,
                    context_embedder=context_embedder,
                    criterion=criterion,
                    optimizer=optimizer,
                    epoch=args.start_epoch)
        return

    best_psnr = 0
    psnr = 0
    for epoch in range(args.start_epoch, args.max_epoch):
        
        # run training

        if args.dataset == 'anime_vectorized' or args.dataset == 'anime_interp':
            train_vectorized(
                args=args,
                train_loader=train_loader,
                model=model,
                vector_model=vector_model,
                svg_encoder=svg_encoder,
                context_embedder=context_embedder,
                criterion=criterion,
                optimizer=optimizer,
                lpips_model=lpips_model,
                epoch=epoch
            )

            test_loss, psnr, _, _ = test_vectorized(
                    args=args,
                    test_loader=test_loader,
                    train_loader=train_loader,
                    model=model,
                    vector_model=vector_model,
                    svg_encoder=svg_encoder,
                    context_embedder=context_embedder,
                    criterion=criterion,
                    optimizer=optimizer,
                    epoch=epoch
                )
        
        else:
            train(
                args=args,
                train_loader=train_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                lpips_model=lpips_model,
                epoch=epoch
                )

            test_loss, psnr, _, _ = test(
                args=args,
                test_loader=test_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                epoch=args.start_epoch
                )
            

        # save checkpoint
        is_best = psnr > best_psnr
        # is_best = True
        best_psnr = max(psnr, best_psnr)
        utils.save_checkpoint({
            'epoch': epoch,
            'cain_state_dict': model.state_dict(),
            'vector_model_state_dict': vector_model.state_dict(),
            'svg_encoder_state_dict': svg_encoder.state_dict(),
            'context_embedder_state_dict': context_embedder.state_dict(), 
            'optimizer': optimizer.state_dict(),
            'best_psnr': best_psnr
        }, is_best, args.exp_name)

        # update optimizer policy
        scheduler.step(test_loss)

if __name__ == "__main__":
    main(args)
