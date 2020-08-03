
# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from dataloader import *
from checkpointer import Checkpointer
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
from model import Model
from tqdm import tqdm
from launcher import run
from net import *
import numpy as np
# import tensorflow as tf


class ImageGenerator:
    def __init__(self, cfg, images, minibatch_gpu):
        self.images = images
        self.minibatch_size = minibatch_gpu
        self.cfg = cfg

    def evaluate(self, logger, model, mapping, decoder, lod, trials=1):
        torch.cuda.set_device(0)
        rnd = np.random.RandomState(None)
        ims = []
        nb = len(self.images)
        for i in tqdm(range(0, nb, self.minibatch_size)):
            X = self.images[i:i+self.minibatch_size].cuda()
            print(X.shape)
            Z, _ = model.encode(X, lod, 1.0)
            Z = Z.repeat(1, model.mapping_fl.num_layers, 1)
            if self.cfg.MODEL.Z_REGRESSION:
                Z = model.mapping_fl(Z[:, 0])
            else:
                Z = Z.repeat(1, model.mapping_fl.num_layers, 1)

            T = []
            for trial in range(trials):
                images = decoder(Z, lod, 1.0, True)
                # Downsample to 256x256. The attribute classifiers were built for 256x256.
                # factor = images.shape[2] // 256
                # if factor != 1:
                    # images = torch.nn.functional.avg_pool2d(images, factor, factor)
                images = np.clip((images.cpu().numpy() + 1.0) * 127, 0, 255).astype(np.uint8)
                T.append(images)
            T = np.array(T)
            T = T.transpose((1,0,2,3,4))
            print(T.shape)
            ims.append(T)
        images = np.concatenate(ims)
        orig = np.clip( (self.images.numpy() + 1.0 ) * 127, 0, 255).astype(np.uint8)
        np.savez("reconstructions.npz", orig=orig, reconstructions=images)

def sample(cfg, logger):
    torch.cuda.set_device(0)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
        truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER)

    model.cuda()
    model.eval()
    model.requires_grad_(False)

    decoder = model.decoder
    encoder = model.encoder
    mapping_tl = model.mapping_tl
    mapping_fl = model.mapping_fl
    dlatent_avg = model.dlatent_avg

    logger.info("Trainable parameters generator:")
    count_parameters(decoder)

    logger.info("Trainable parameters discriminator:")
    count_parameters(encoder)

    arguments = dict()
    arguments["iteration"] = 0

    model_dict = {
        'discriminator_s': encoder,
        'generator_s': decoder,
        'mapping_tl_s': mapping_tl,
        'mapping_fl_s': mapping_fl,
        'dlatent_avg': dlatent_avg
    }

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                {},
                                logger=logger,
                                save=False)

    checkpointer.load()

    model.eval()

    layer_count = cfg.MODEL.LAYER_COUNT

    logger.info("Generating...")

    decoder = nn.DataParallel(decoder)
    mapping_fl = nn.DataParallel(mapping_fl)
    
    
    xlist = []
    lod = 7
    resolution = (2**lod) * 4
    pattern = "example_images/*.jpg"
    from glob import glob
    from imageio import imread
    from skimage.transform import resize
    print(resolution)
    for path in glob(pattern):
        x = imread(path)
        x = resize(x, (resolution, resolution), preserve_range=True)
        x = x.astype("float32")
        print(x.min(), x.max())
        x = x / 127.5 - 1
        x = x.transpose((2,0,1))
        xlist.append(x)
    images = np.array(xlist)
    images = torch.from_numpy(images)
    images = images.float()
    print(images.shape)
    with torch.no_grad():
        gen = ImageGenerator(cfg, images, minibatch_gpu=8)
        gen.evaluate(logger, model, mapping_fl, decoder, lod, trials=5)


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='ALAE-generate-images-for-attribute-classifications',
        default_config='configs/ffhq.yaml',
        world_size=gpu_count, write_log=False)
