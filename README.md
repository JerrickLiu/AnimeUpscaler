# AnimeUpscaler

AnimeUpscaler is a deep neural network trained to interpolate animation frames of cartoon and anime. It's based on both rasterized and vectorized data. The model architecture is based on the paper [Channel Attention is All You Need for Video Frame Interpolation](https://github.com/myungsub/CAIN).


# Directory structure
```
project
   vtracer
      | Scripts needed to vectorize data
   models
      CAIN
        │   README.md
        |   run.sh - main script to train CAIN model
        |   run_noca.sh - script to train CAIN_NoCA model
        |   test_custom.sh - script to run interpolation on custom dataset
        |   eval.sh - script to evaluate on SNU-FILM benchmark
        |   main.py - main file to run train/val
        |   config.py - check & change training/testing configurations here
        |   loss.py - defines different loss functions
        |   utils.py - misc.
        └───model
        │   │   common.py
        │   │   cain.py - main model
        |   |   cain_noca.py - model without channel attention
        |   |   cain_encdec.py - model with additional encoder-decoder
        └───data - implements dataloaders for each dataset
        │   |   vimeo90k.py - main training / testing dataset
        |   |   video.py - custom data for testing
        │   └───symbolic links to each dataset
        |       | ...
      AnimeInterp
        |  Code for AnimeInterp paper
```
