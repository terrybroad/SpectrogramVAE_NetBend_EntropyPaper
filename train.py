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
from model import Encoder, Decoder

torch.set_default_tensor_type('torch.cuda.FloatTensor')

#Waveform to Spectrogram conversion

''' Decorsière, Rémi, Peter L. Søndergaard, Ewen N. MacDonald, and Torsten Dau.
"Inversion of auditory spectrograms, traditional spectrograms, and other envelope representations."
IEEE/ACM Transactions on Audio, Speech, and Language Processing 23, no. 1 (2014): 46-56.'''

#ORIGINAL CODE FROM https://github.com/yoyololicon/spectrogram-inversion

# melobj = MelScale(n_mels=hop, sample_rate=sr, f_min=0.)
# melfunc = melobj.forward

# def melspecfunc(waveform):
#   specgram = specfunc(waveform)
#   mel_specgram = melfunc(specgram)
#   return mel_specgram

def spectral_convergence(input, target):
    return 20 * ((input - target).norm().log10() - target.norm().log10())

def GRAD(spec, transform_fn, samples=None, init_x0=None, maxiter=1000, tol=1e-6, verbose=1, evaiter=10, lr=0.002):

    spec = torch.Tensor(spec)
    samples = (spec.shape[-1]*hop)-hop

    if init_x0 is None:
        init_x0 = spec.new_empty((1,samples)).normal_(std=1e-6)
    x = nn.Parameter(init_x0)
    T = spec

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam([x], lr=lr)

    bar_dict = {}
    metric_func = spectral_convergence
    bar_dict['spectral_convergence'] = 0
    metric = 'spectral_convergence'

    init_loss = None
    with tqdm(total=maxiter, disable=not verbose) as pbar:
        for i in range(maxiter):
            optimizer.zero_grad()
            V = transform_fn(x)
            loss = criterion(V, T)
            loss.backward()
            optimizer.step()
            lr = lr*0.9999
            for param_group in optimizer.param_groups:
              param_group['lr'] = lr

            if i % evaiter == evaiter - 1:
                with torch.no_grad():
                    V = transform_fn(x)
                    bar_dict[metric] = metric_func(V, spec).item()
                    l2_loss = criterion(V, spec).item()
                    pbar.set_postfix(**bar_dict, loss=l2_loss)
                    pbar.update(evaiter)

    return x.detach().view(-1).cpu()

def normalize(S, args):
  return np.clip((((S - args.min_db) / -args.min_db)*2.)-1., -1, 1)

def denormalize(S, args):
  return (((np.clip(S, -1, 1)+1.)/2.) * -args.min_db) + args.min_db

def prep(wv, args, specfunc):
  S = np.array(torch.squeeze(specfunc(torch.Tensor(wv).view(1,-1))).detach().cpu())
  S = librosa.power_to_db(S)-args.ref_db
  return normalize(S, args)

def deprep(S, args, specfunc):
  S = denormalize(S)+args.ref_db
  S = librosa.db_to_power(S)
  wv = GRAD(np.expand_dims(S,0), specfunc, maxiter=2500, evaiter=10, tol=1e-8)
  return np.array(np.squeeze(wv))

#---------Helper functions------------#

#Generate spectrograms from waveform array
def tospec(data, args, specfunc):
  specs=np.empty(data.shape[0], dtype=object)
  for i in range(data.shape[0]):
    x = data[i]
    S=prep(x, args, specfunc)
    S = np.array(S, dtype=np.float32)
    specs[i]=np.expand_dims(S, -1)
  print(specs.shape)
  return specs

#Generate multiple spectrograms with a determined length from single wav file
def tospeclong(path, length=4*44100):
  x, sr = librosa.load(path,sr=44100)
  x,_ = librosa.effects.trim(x)
  loudls = librosa.effects.split(x, top_db=50)
  xls = np.array([])
  for interv in loudls:
    xls = np.concatenate((xls,x[interv[0]:interv[1]]))
  x = xls
  num = x.shape[0]//length
  specs=np.empty(num, dtype=object)
  for i in range(num-1):
    a = x[i*length:(i+1)*length]
    S = prep(a)
    S = np.array(S, dtype=np.float32)
    try:
      sh = S.shape
      specs[i]=S
    except AttributeError:
      print('spectrogram failed')
  print(specs.shape)
  return specs

#Waveform array from path of folder containing wav files
def audio_array(path):
  ls = glob(f'{path}/*.wav')
  adata = []
  for i in range(len(ls)):
    print(ls[i])
    x, sr = torchaudio.load(ls[i])
    x = x.numpy()
    print(x)
    adata.append(x[0])
  return np.array(adata)

#Concatenate spectrograms in array along the time axis
def testass(a):
  but=False
  con = np.array([])
  nim = a.shape[0]
  for i in range(nim):
    im = a[i]
    im = np.squeeze(im)
    if not but:
      con=im
      but=True
    else:
      con = np.concatenate((con,im), axis=1)
  return np.squeeze(con)

#Split spectrograms in chunks with equal size
def splitcut(data,args):
  ls = []
  mini = 0
  minifinal = args.spec_split*args.shape   #max spectrogram length
  for i in range(data.shape[0]-1):
    if data[i].shape[1]<=data[i+1].shape[1]:
      mini = data[i].shape[1]
    else:
      mini = data[i+1].shape[1]
    if mini>=3*args.shape and mini<minifinal:
      minifinal = mini
  for i in range(data.shape[0]):
    x = data[i]
    if x.shape[1]>=3*args.shape:
      for n in range(x.shape[1]//minifinal):
        ls.append(x[:,n*minifinal:n*minifinal+minifinal,:])
      ls.append(x[:,-minifinal:,:])
  return np.array(ls)


if __name__ == "__main__":
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--vector_dim', type=int, default=128)
    parser.add_argument('--maxiter', type=int, default=100001)
    parser.add_argument('--hop', type=int, default=256)
    parser.add_argument('--sr', type=int, default=44100)
    parser.add_argument('--min_db', type=int, default=-100)
    parser.add_argument('--ref_db', type=int, default=20)
    parser.add_argument('--spec_split', type=int, default=1)
    parser.add_argument('--shape', type=int, default=128)
    parser.add_argument('--beta', type=int, default=0.001)
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

    #AUDIO TO CONVERT
    awv = audio_array(args.data)         #get waveform array from folder containing wav files
    print(awv)
    aspec = tospec(awv, args, specfunc)                        #get spectrogram array
    adata = splitcut(aspec, args)                    #split spectrogams to fixed
    print(np.shape(adata))

    number_of_rows = adata.shape[0]
    with tqdm(total=args.maxiter) as pbar:
        for i in range(args.maxiter):
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

            loss = recon_loss + kld * args.beta
            print("iter: "+str(i)+ ", total_loss: "+str(loss.item())+", recon_loss: " + str(recon_loss.item()) + ", kld: "+str(kld.item()))
            loss.backward()
            e_optim.step()
            d_optim.step()

            if i % 100 == 0:
              print("howdy")
              utils.save_image(x, f'sample/{str(i).zfill(6)}_input.png',
                nrow=8,
                normalize=True,
                range=(-1, 1))
              # save_spec_as_image(x[:,0], f'sample/{str(i).zfill(6)}_skspec.png')
              utils.save_image(_x, f'sample/{str(i).zfill(6)}_output.png',
                nrow=8,
                normalize=True,
                range=(-1, 1))

            if i % 1000 == 0:
              torch.save(
                {
                  "encoder": encoder.state_dict(),
                  "decoder": decoder.state_dict(),
                  "e_optim": e_optim.state_dict(),
                  "d_optim": d_optim.state_dict()
                }, 
                args.save_dir+'/checkpoint_'+str(i)+'.pt')    

