import os
import sys
import time
import copy
import shutil
import random

import torch
import numpy as np
from tqdm import tqdm

import config
import utils
import cv2

from data import AnimeDataset, AnimeVectorizedDataset, AnimeInterpDataset

from svg_encoder import *
from data.render_segments import *


##### Parse CmdLine Arguments #####
args, unparsed = config.get_args()
cwd = os.getcwd()
print(args)

def make_video(original_video_path, frame_folder_path, video_name):
    images = [img for img in sorted(os.listdir(frame_folder_path)) if img.endswith(".png") or img.endswith(".jpg")]

    frame = cv2.imread(os.path.join(frame_folder_path, images[0]))
    height, width, layers = frame.shape

    videoCapture = cv2.VideoCapture(original_video_path)
    fps = int(videoCapture.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    video = cv2.VideoWriter(video_name, fourcc, fps * 4, (width,height))

    for image in sorted(images):
        video.write(cv2.imread(os.path.join(frame_folder_path, image)))

    video.release()


def load_dataset(args):
    """ 
    Load the dataset. If the dataset is the anime dataset, use our custom dataset class.
    """

    # TODO: Make this more robust
    if args.dataset == 'anime':
        test_loader = AnimeDataset.get_loader('', args.test_data_root + '/metadata/all_scenes.csv', args.test_data_root + '/extracted_frames', args.batch_size, True, args.num_workers, args.test_mode)

    elif args.dataset == 'anime_vectorized':
        from data.video import get_vectorized_loader
        test_loader = get_vectorized_loader('test', args.test_data_root, args.test_svg_dir, args.test_batch_size, img_fmt=args.img_fmt, shuffle=False)

    elif args.dataset == 'anime_interp':
        test_loader = AnimeInterpDataset.get_loader('', args.test_data_root, args.test_svg_dir, args.batch_size, True, args.num_workers, args.test_mode)

    else:
        test_loader = utils.load_dataset(
            args.dataset, args.test_data_root, args.batch_size, args.test_batch_size, args.num_workers, args.test_mode)
    return test_loader


device = torch.device('cuda' if args.cuda else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

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


def test(args, model, epoch):
    print('Evaluating for epoch = %d' % epoch)
    ##### Load Dataset #####
    test_loader = utils.load_dataset(
        args.dataset, args.test_data_root, args.batch_size, args.test_batch_size, args.num_workers, img_fmt=args.img_fmt)
    model.eval()

    t = time.time()
    with torch.no_grad():
        for i, (images, meta) in enumerate(tqdm(test_loader)):

            print(meta['imgpath'][1])

            # Build input batch
            im1, im2 = images[0].to(device), images[1].to(device)

            # Forward
            out, _ = model(im1, im2)

            # Save result images
            if args.mode == 'test':
                for b in range(images[0].size(0)):
                    paths = meta['imgpath'][0][b].split('/')
                    fp = args.test_data_root
                    fp = os.path.join(fp, paths[-1][:-4])   # remove '.png' extension
                    
                    # Decide float index
                    i1_str = paths[-1][:-4]
                    i2_str = meta['imgpath'][1][b].split('/')[-1][:-4]
                    try:
                        i1 = float(i1_str.split('_')[-1])
                    except ValueError:
                        i1 = 0.0
                    try:
                        i2 = float(i2_str.split('_')[-1])
                        if i2 == 0.0:
                            i2 = 1.0
                    except ValueError:
                        i2 = 1.0

                    fpos = max(0, fp.rfind('_'))

                    fInd = i1 + i2 / 2
                    savepath = "%s_%06f.%s" % (fp[:fpos], fInd, args.img_fmt)

                    utils.save_image(out[b], savepath)
                    
    # Print progress
    print('im_processed: {:d}/{:d} {:.3f}s   \r'.format(i + 1, len(test_loader), time.time() - t))

    return

def test_vectorized(args, test_loader, model, vector_model, svg_encoder, context_embedder, epoch):
    print('Evaluating for epoch = %d' % epoch)

    model.eval()
    vector_model.eval()
    svg_encoder.eval()
    context_embedder.eval()

    t = time.time()
    with torch.no_grad():
        for i, (images, meta, svgs, num_segments, svg_files, svg_prepad_info) in enumerate(tqdm(test_loader)):

            # Build input batch
            images, svgs = images.to(device), svgs.to(device)

            im1 = images[:, 0, ...]
            im2 = images[:, 1, ...]

            context_vectors = embed_svgs(svgs, svg_encoder, context_embedder)
            norms = torch.norm(context_vectors, dim=2, keepdim=True)
            context_normed = context_vectors / norms
            frame1_batch = context_normed[0::2, :, :]
            frame3_batch = context_normed[1::2, :, :]
            sim = torch.bmm(frame1_batch, frame3_batch.transpose(1, 2))

            masks = batch_render_clusters_correspondence(svg_files, svg_prepad_info, sim, num_segments)


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
            out, _ = model(im1, im2, intermediate)

            # Save result images
            if args.mode == 'test':
                for b in range(images[0].size(0)):
                    paths = meta['imgpath'][0][b].split('/')
                    fp = args.test_data_root
                    fp = os.path.join(fp, paths[-1][:-4])   # remove '.png' extension
                    
                    # Decide float index
                    i1_str = paths[-1][:-4]
                    i2_str = meta['imgpath'][1][b].split('/')[-1][:-4]
                    try:
                        i1 = float(i1_str.split('_')[-1])
                    except ValueError:
                        i1 = 0.0
                    try:
                        i2 = float(i2_str.split('_')[-1])
                        if i2 == 0.0:
                            i2 = 1.0
                    except ValueError:
                        i2 = 1.0

                    fpos = max(0, fp.rfind('_'))

                    fInd = i1 + i2 / 2
                    savepath = "%s_%06f.%s" % (fp[:fpos], fInd, args.img_fmt)
                    utils.save_image(out[b], savepath)
                    
    # Print progress
    print('im_processed: {:d}/{:d} {:.3f}s   \r'.format(i + 1, len(test_loader), time.time() - t))

    return

""" Entry Point """
def main(args):

    model, vector_model, svg_encoder, context_embedder = build_model(args)

    ##### Load Dataset #####
    test_loader = load_dataset(args)

    # If resume, load checkpoint: model
    if args.resume:
        #utils.load_checkpoint(args, model, optimizer=None)
        print("Loading checkpoint...")
        checkpoint = torch.load(args.checkpoint_path)
        args.start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['cain_state_dict'])
        vector_model.load_state_dict(checkpoint['vector_model_state_dict'])
        svg_encoder.load_state_dict(checkpoint['svg_encoder_state_dict'])
        context_embedder.load_state_dict(checkpoint['context_embedder_state_dict'])
        del checkpoint
        print("Loaded!")

    num_iter = 2 # x2**num_iter interpolation
    for _ in range(num_iter):
        
        # run test
        test_vectorized(args, test_loader, model, vector_model, svg_encoder, context_embedder, args.start_epoch)
        # test(args, model, args.start_epoch)

    make_video(args.orig_video_path, args.data_root, args.save_video_path)

if __name__ == "__main__":
    main(args)
