import os
import math
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
from torch.utils.data import DataLoader
from model import Encoder, Decoder
from util import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')

if __name__ == "__main__":
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--vector_dim', type=int, default=128)
    parser.add_argument('--iter_p_batch', type=int, default=100)
    parser.add_argument('--tracks_p_batch', type=int, default=10)
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
    parser.add_argument('--run_name', type=str, default="test")
    parser.add_argument('--save_dir', type=str, default="ckpt")
    args = parser.parse_args()

    encoder = Encoder(args.vector_dim)
    decoder = Decoder(args.vector_dim)

    e_optim = optim.Adam(encoder.parameters(), lr=args.lr, betas=(0, 0.99))
    d_optim = optim.Adam(decoder.parameters(), lr=args.lr, betas=(0, 0.99))
    criterion = nn.MSELoss()

    if args.ckpt != "":
      state_dict = torch.load(args.ckpt)
      encoder.load_state_dict(state_dict['encoder'])
      decoder.load_state_dict(state_dict['decoder'])
      e_optim.load_state_dict(state_dict['e_optim'])
      d_optim.load_state_dict(state_dict['d_optim'])

    specobj = Spectrogram(n_fft=4*args.hop, win_length=4*args.hop, hop_length=args.hop, pad=0, power=2, normalized=False)
    specfunc = specobj.forward


    dataset = AudioData(args.data)

    dataloader = DataLoader(dataset, batch_size=args.tracks_p_batch, collate_fn=collate_list, shuffle=True, num_workers=0)

    it_count = 0
    
    with tqdm(total=args.num_epochs) as pbar:
      
      for i in range(args.num_epochs):
        
        for index, sample in enumerate(dataloader):
          
          awv = audio_array_from_batch(sample)
          aspec = tospec(awv, args, specfunc)                        #get spectrogram array
          adata = splitcut(aspec, args)      

          for j in range(args.iter_p_batch):
              e_optim.zero_grad()
              d_optim.zero_grad()

              batch = adata[np.random.randint(adata.shape[0], size=args.batch), :]
              
              x = torch.tensor(batch).to('cuda').transpose(1,3)
              z, kld = encoder(x)
              _x = decoder(z)

              recon_loss = criterion(x, _x)

              #TAKE THE LOG TO TRY AND OVERCOME POSTERIOR COLLAPSE
              #ADD ONE TO AVOID NEGATIVE NUMBERS
              kld = torch.log(kld + torch.tensor(1).detach())

              if math.isinf(kld.item()):
                kld = torch.Tensor([[0]]).cuda().detach()

              loss = recon_loss + kld * args.beta
              print("epoch: "+ str(i) + ", iter: "+str(it_count)+ ", total_loss: "+str(loss.item())+", recon_loss: " + str(recon_loss.item()) + ", kld: "+str(kld.item()))
              loss.backward()
              e_optim.step()
              d_optim.step()

              if it_count % 100 == 0:
                print("howdy")
                utils.save_image(x, f'sample/{str(it_count).zfill(6)}_input.png',
                  nrow=4,
                  normalize=True,
                  range=(-1, 1))
                # save_spec_as_image(x[:,0], f'sample/{str(i).zfill(6)}_skspec.png')
                utils.save_image(_x, f'sample/{str(it_count).zfill(6)}_output.png',
                  nrow=4,
                  normalize=True,
                  range=(-1, 1))

              if it_count % 10000 == 0:
                torch.save(
                  {
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "e_optim": e_optim.state_dict(),
                    "d_optim": d_optim.state_dict()
                  }, 
                  args.save_dir+'/checkpoint_'+str(it_count)+'.pt')    

              it_count += 1

