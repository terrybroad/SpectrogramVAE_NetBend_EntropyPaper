import os
import math
import util
import yaml
import heapq
import torch
import argparse
import torchaudio
import librosa
import librosa.display
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from torch.utils.data import DataLoader

from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from numpy import linspace
from glob import glob
from torchvision import utils
from torch import autograd, optim
from tqdm import tqdm
from functools import partial
from torchaudio.transforms import MelScale, Spectrogram
from model import Encoder, Decoder
from util import *
from transform_util import *

# torch.set_default_tensor_type('torch.cuda.FloatTensor')

def generate_activation_training_set(encoder, decoder, args, specfunc, data, transform_dict_list):
  with torch.no_grad():
    for index, sample in enumerate(dataloader):
        if index < 1000:
            awv = audio_array_from_batch(sample)
            aspec = tospec(awv, args, specfunc)                        #get spectrogram array
            adata = splitcut(aspec, args)       
            for j in range(args.iter_p_batch):
                extra_t_dict_list = []
                extra_t_dict_list.append({'layerID': -1, 'index': index, 'params': [1, 4]})
                batch = adata[np.random.randint(adata.shape[0], size=args.batch), :]
                x = torch.tensor(batch).to('cuda').transpose(1,3)
                z, kld, mu = encoder(x)
                # sample_z = torch.randn(1, args.vector_dim, device=device)
                sample = decoder(mu, extra_t_dict_list)
                if not os.path.exists('activations/input_im'):
                        os.makedirs('activations/input_im')
                utils.save_image(
                    x[0,:],
                    f'activations/input_im/{str(index).zfill(6)}.png',
                    nrow=1,
                    normalize=True,
                    range=(-1, 1))
                if not os.path.exists('activations/output_im'):
                        os.makedirs('activations/output_im')
                utils.save_image(
                    sample[0,:],
                    f'activations/output_im/{str(index).zfill(6)}.png',
                    nrow=1,
                    normalize=True,
                    range=(-1, 1))


if __name__ == "__main__":
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--vector_dim', type=int, default=128)
    parser.add_argument('--iter_p_batch', type=int, default=1)
    parser.add_argument('--tracks_p_batch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--hop', type=int, default=256)
    parser.add_argument('--sr', type=int, default=44100)
    parser.add_argument('--min_db', type=int, default=-100)
    parser.add_argument('--ref_db', type=int, default=20)
    parser.add_argument('--spec_split', type=int, default=1)
    parser.add_argument('--shape', type=int, default=128)
    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--ckpt', type=str, default="")
    parser.add_argument('--data', type=str, default="/home/terence/Music/bach_wavs")
    parser.add_argument('--clusters', type=str, default="")
    parser.add_argument('--noise', type=bool, default=False)
    parser.add_argument('--config', type=str, default="configs/example_transform_config.yaml")

    args = parser.parse_args()

    encoder = Encoder(args.vector_dim)
    decoder = Decoder(args.vector_dim)

    e_optim = optim.Adam(encoder.parameters(), lr=args.lr, betas=(0, 0.99))
    d_optim = optim.Adam(decoder.parameters(), lr=args.lr, betas=(0, 0.99))
    criterion = nn.MSELoss()

    yaml_config = {}
    with open(args.config, 'r') as stream:
        try:
            yaml_config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    cluster_config = {}
    if args.clusters != "":
        with open(args.clusters, 'r') as stream:
            try:
                cluster_config = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    transform_dict_list = create_transforms_dict_list(yaml_config, cluster_config, layer_channel_dims)
    
    checkpoint = torch.load(args.ckpt)

    new_state_dict_e = encoder.state_dict()
    new_state_dict_e.update(checkpoint['encoder'])
    encoder.load_state_dict(new_state_dict_e)

    new_state_dict_d = decoder.state_dict()
    new_state_dict_d.update(checkpoint['decoder'])
    decoder.load_state_dict(new_state_dict_d)

    encoder.to('cuda')
    decoder.to('cuda')

    dataset = AudioData(args.data)
    dataloader = DataLoader(dataset, batch_size=args.tracks_p_batch, collate_fn=collate_list, shuffle=True, num_workers=0)

    specobj = Spectrogram(n_fft=4*args.hop, win_length=4*args.hop, hop_length=args.hop, pad=0, power=2, normalized=False)
    specfunc = specobj.forward

    generate_activation_training_set(encoder, decoder, args, specfunc, dataloader, transform_dict_list)