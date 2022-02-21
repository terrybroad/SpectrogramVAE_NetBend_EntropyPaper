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
from transform_util import *
from clustering_models import FeatureClassifier
from tqdm import tqdm
from util import *
from kmeans_pytorch import kmeans

cluster_layer_dict = {
    1 : 5,
    2 : 5,
    3 : 4,
    4 : 4
}

def get_clusters_from_generated_average(args, encoder, decoder, specfunc, dataloader, t_dict_list, layer_channel_dims):
    print("get clusters")
    with torch.no_grad():
        latent_ll = []
        feature_ll = []
        feature_cluster_sum_dict = {}
        feature_cluster_dict = {}
        feature_latent_dict = {}
        
        for i in tqdm(range(args.n_layers)):
            print("I" + str(i))
            true_index = i+1
            latent_list = []
            feature_list = []
            latent_ll.append(latent_list)
            feature_ll.append(feature_list)
            feature_cluster_sum_dict[true_index] = {}
            for j in tqdm(range(layer_channel_dims[true_index])):
                feature_cluster_sum_dict[true_index][j] = 0 
                latent_ll[i].append(0)

        for index, sample in enumerate(dataloader):
            if index < args.num_samples:
                awv = audio_array_from_batch(sample)
                aspec = tospec(awv, args, specfunc)                        #get spectrogram array
                adata = splitcut(aspec, args)       
                for j in range(args.iter_p_batch):
                    print("processing sample: " + str(index))
                    extra_t_dict_list = []
                    extra_t_dict_list.append({'layerID': -1, 'index': index, 'params': [1, 4]})
                    batch = adata[np.random.randint(adata.shape[0], size=args.batch), :]
                    x = torch.tensor(batch).to('cuda').transpose(1,3)
                    z, kld, mu = encoder(x)
                    sample, activation_maps = decoder(mu, extra_t_dict_list,return_activation_maps=True)
                    for index, activations in enumerate(activation_maps):
                        true_index = index+1
                        classifier = FeatureClassifier(true_index)
                        classifier_str = args.classifier_ckpts + "/" + str(true_index) + "/classifier" + str(true_index) + "_final.pt"
                        classifier_state_dict = torch.load(classifier_str)
                        classifier.load_state_dict(classifier_state_dict)
                        classifier.to(device)
                        layer_activation_maps = activation_maps[index]
                        a_map_array = list(torch.split(layer_activation_maps,1,1))
                        for j, map in enumerate(a_map_array):                  
                            map = map.to(device)
                            feat_vec, class_prob = classifier(map)
                            normalised_feat_vec = feat_vec[0,:] / args.num_samples
                            latent_ll[index][j] = latent_ll[index][j] + normalised_feat_vec
                            # feature_ll[index].append(j)
            else:
                break
        
        for i in tqdm(range(args.n_layers)):
            true_index = i+1
            print("generating clusters for layer: " + str(i))
            x = torch.stack(latent_ll[i])
            print(x.shape)
            cluster_ids_x, cluster_centers = kmeans(X=x, num_clusters=cluster_layer_dict[true_index], distance='euclidean', device=torch.device('cuda'))
            dict_list = []
            latent_dict_list = []
            for j, id in enumerate(cluster_ids_x):
                cluster_dict = {"feature_index": int(j), "cluster_index": int(id)}
                latent_dict = {"feature_index": int(j), "latent": latent_ll[i][j].to('cpu').numpy().tolist()}
                dict_list.append(cluster_dict)
                latent_dict_list.append(latent_dict)
            feature_cluster_dict[true_index] = dict_list
            feature_latent_dict[true_index] = latent_dict_list
    
    with open(r'cluster_dict.yaml', 'w') as file:
        documents = yaml.dump(feature_cluster_dict, file)
    with open(r'latent_dict.yaml', 'w') as file:
        documents = yaml.dump(feature_latent_dict, file)


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--vector_dim', type=int, default=128)
    parser.add_argument('--iter_p_batch', type=int, default=1)
    parser.add_argument('--tracks_p_batch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--hop', type=int, default=256)
    parser.add_argument('--sr', type=int, default=44100)
    parser.add_argument('--min_db', type=int, default=-100)
    parser.add_argument('--ref_db', type=int, default=20)
    parser.add_argument('--spec_split', type=int, default=1)
    parser.add_argument('--shape', type=int, default=128)
    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--ckpt', type=str, default="")
    parser.add_argument('--data', type=str, default="/home/terence/Music/bach_wavs")
    parser.add_argument('--noise', type=bool, default=False)
    parser.add_argument('--config', type=str, default="configs/example_transform_config.yaml")
    parser.add_argument('--classifier_ckpts', type=str, default="models/classifiers")
    parser.add_argument('--n_layers', type=int, default=4)

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
    
    # layer_channel_dims = create_layer_channel_dim_dict(args.channel_multiplier, args.n_layers)
    transform_dict_list = create_transforms_dict_list(yaml_config, {}, layer_channel_dims)

    get_clusters_from_generated_average(args, encoder, decoder, specfunc, dataloader, transform_dict_list, layer_channel_dims)