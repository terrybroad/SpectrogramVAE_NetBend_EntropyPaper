import tensorflow as tf
import os

import util


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from glob import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from numpy import linspace
import soundfile as sf
from torchvision import utils

#Hyperparameters
LEARNING_RATE = 0.0005
EPOCHS =  40
BATCH_SIZE  = 64 
VECTOR_DIM = 128

hop=256               #hop size (window size = 4*hop)
sr=44100              #sampling rate
min_level_db=-100     #reference values to normalize data
ref_level_db=20

shape=128           #length of time axis of split specrograms
spec_split=1

maxiter=100000

#Waveform to Spectrogram conversion

''' Decorsière, Rémi, Peter L. Søndergaard, Ewen N. MacDonald, and Torsten Dau.
"Inversion of auditory spectrograms, traditional spectrograms, and other envelope representations."
IEEE/ACM Transactions on Audio, Speech, and Language Processing 23, no. 1 (2014): 46-56.'''

#ORIGINAL CODE FROM https://github.com/yoyololicon/spectrogram-inversion

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd, optim
from tqdm import tqdm
from functools import partial
import math
import heapq
from torchaudio.transforms import MelScale, Spectrogram
from model import Encoder, Decoder, Discriminator

torch.set_default_tensor_type('torch.cuda.FloatTensor')

specobj = Spectrogram(n_fft=4*hop, win_length=4*hop, hop_length=hop, pad=0, power=2, normalized=False)
specfunc = specobj.forward
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

def normalize(S):
  return np.clip((((S - min_level_db) / -min_level_db)*2.)-1., -1, 1)

def denormalize(S):
  return (((np.clip(S, -1, 1)+1.)/2.) * -min_level_db) + min_level_db

def prep(wv,hop=192):
  S = np.array(torch.squeeze(specfunc(torch.Tensor(wv).view(1,-1))).detach().cpu())
  S = librosa.power_to_db(S)-ref_level_db
  return normalize(S)

def deprep(S):
  S = denormalize(S)+ref_level_db
  S = librosa.db_to_power(S)
  wv = GRAD(np.expand_dims(S,0), specfunc, maxiter=2500, evaiter=10, tol=1e-8)
  return np.array(np.squeeze(wv))

#---------Helper functions------------#

#Generate spectrograms from waveform array
def tospec(data):
  specs=np.empty(data.shape[0], dtype=object)
  for i in range(data.shape[0]):
    x = data[i]
    S=prep(x)
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

# #Waveform array from path of folder containing wav files
# def audio_array(path):
#   ls = glob(f'{path}/*.wav')
#   adata = []
#   for i in range(len(ls)):
#     #CHANGE TO TORCHAUDIO
#     x, sr = tf.audio.decode_wav(tf.io.read_file(ls[i]), 1)
#     x = np.array(x, dtype=np.float32)
#     adata.append(x)
#   return np.array(adata)

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
def splitcut(data):
  ls = []
  mini = 0
  minifinal = spec_split*shape   #max spectrogram length
  for i in range(data.shape[0]-1):
    if data[i].shape[1]<=data[i+1].shape[1]:
      mini = data[i].shape[1]
    else:
      mini = data[i+1].shape[1]
    if mini>=3*shape and mini<minifinal:
      minifinal = mini
  for i in range(data.shape[0]):
    x = data[i]
    if x.shape[1]>=3*shape:
      for n in range(x.shape[1]//minifinal):
        ls.append(x[:,n*minifinal:n*minifinal+minifinal,:])
      ls.append(x[:,-minifinal:,:])
  return np.array(ls)

def save_spec_as_image(spectrogram, out_path):

    # spec = numpy.log(spectrogram + 1e-9) # add small number to avoid log(0)
    spec = Spectrogram.to('cpu').to
    # min-max scale to fit inside 8-bit range
    img = scale_minmax(spec, 0, 255).astype(numpy.uint8)
    img = numpy.flip(img, axis=1) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy
    # save as PNG
    skimage.io.imsave(out_path, img)

"""## Training"""

#Import folder containing .wav files for training
#Generating Mel-Spectrogram dataset (Uncomment where needed)
#adata: source spectrograms

audio_directory = "/home/terence/Music/bach_wavs"

#AUDIO TO CONVERT
awv = audio_array(audio_directory)         #get waveform array from folder containing wav files
aspec = tospec(awv)                        #get spectrogram array
adata = splitcut(aspec)                    #split spectrogams to fixed
print(np.shape(adata))

#Start training from scratch or resume training

training_run_name = "aerofonos_test_train" 
checkpoint_save_directory = ""
resume_training = False 
resume_training_checkpoint_path = "" 
# current_time = get_time_stamp()

encoder = Encoder(128)
generator = Decoder(128)
discriminator = Discriminator(128)

beta = 0.2

e_optim = optim.Adam(encoder.parameters(), lr=0.0003, betas=(0, 0.99))
g_optim = optim.Adam(generator.parameters(), lr=0.0003, betas=(0, 0.99))
d_optim = optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0, 0.99))

distance = torch.nn.PairwiseDistance()
criterion = nn.MSELoss()

if __name__ == "__main__":
    number_of_rows = adata.shape[0]
    torch.autograd.set_detect_anomaly(True)
    with tqdm(total=maxiter) as pbar:
        for i in range(maxiter):
            encoder.zero_grad()
            discriminator.zero_grad()
            generator.zero_grad()

            e_optim.zero_grad()
            g_optim.zero_grad()
            d_optim.zero_grad()

            util.requires_grad(generator, True)
            util.requires_grad(encoder, True)
            util.requires_grad(discriminator, False)

            batch = adata[np.random.randint(adata.shape[0], size=BATCH_SIZE), :]
            x = torch.tensor(batch).to('cuda').transpose(1,3)
            z, kld = encoder(x)
            reconstruction = generator(z)

            gen_z = torch.randn(BATCH_SIZE, 128, device='cuda' )
            fake = generator(gen_z)

            real_enc, real_prob = discriminator(x)
            rec_enc, rec_prob  = discriminator(reconstruction)
            fake_enc, fake_prob = discriminator(fake)

            # disc_distance = distance(real_enc, rec_enc)
            recon_loss = criterion(x,reconstruction)
            gen_adv_loss = (F.softplus(-rec_prob).mean() + F.softplus(-fake_prob).mean())/2
            
            vae_loss = recon_loss + beta * kld.mean() + gen_adv_loss
            vae_loss.backward()
            e_optim.step()
            g_optim.step()

            encoder.zero_grad()
            discriminator.zero_grad()
            generator.zero_grad()

            e_optim.zero_grad()
            g_optim.zero_grad()
            d_optim.zero_grad()

            util.requires_grad(generator, False)
            util.requires_grad(encoder, False)
            util.requires_grad(discriminator, True)

            batch = adata[np.random.randint(adata.shape[0], size=BATCH_SIZE), :]
            x = torch.tensor(batch).to('cuda').transpose(1,3)
            z, kld = encoder(x)
            reconstruction = generator(z)

            gen_z = torch.randn(BATCH_SIZE, 128, device='cuda' )
            fake = generator(gen_z)

            real_enc, real_prob = discriminator(x)
            rec_enc, rec_prob  = discriminator(reconstruction)
            fake_enc, fake_prob = discriminator(fake)


            disc_loss = F.softplus(-real_prob).mean() + F.softplus(rec_prob).mean() + F.softplus(fake_prob).mean()
            disc_loss.backward()

            d_optim.step()

            print("iter: "+str(i)+ ", vae_loss: "+str(vae_loss.item())+", recon_loss: " + str(recon_loss.mean().item()) + ", kld: "+str(kld.item())  + ", disc_loss: "+str(disc_loss.item()) + ", gen_adv_loss: "+str(gen_adv_loss.item()))
 

            if i % 100 == 0:
              print("howdy")
              utils.save_image(x, f'sample/{str(i).zfill(6)}_input.png',
                nrow=8,
                normalize=True,
                range=(-1, 1))
              utils.save_image(fake, f'sample/{str(i).zfill(6)}_output.png',
                nrow=8,
                normalize=True,
                range=(-1, 1))

            if i % 1000 == 0:
              torch.save(
                {
                  "encoder": encoder.state_dict(),
                  "generator": generator.state_dict(),
                  "discriminator": discriminator.state_dict(),
                  "e_optim": e_optim.state_dict(),
                  "g_optim": d_optim.state_dict(),
                  "d_optim": g_optim.state_dict
                }, 
                'checkpoint'+str(i)+'.pt')    

