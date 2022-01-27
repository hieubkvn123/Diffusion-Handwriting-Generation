import nn
import utils
import preprocessing

import os
import tqdm
import numpy as np
import tensorflow as tf

class HandWritingGenerator:
    def __init__(self, weights_path, writer_source, num_attlayers=2, channels=128):
        self.weights_path = weights_path
        self.writer_source = writer_source
        self.num_attlayers = num_attlayers
        self.channels = channels

        if(not os.path.exists(self.weights_path)):
            raise Exception("Model weights path do not exist")

        if(not os.path.exists(self.writer_source)):
            raise Exception("Writer source path do not exist")

        # Initialize the model
        self.tokenizer = utils.Tokenizer()
        self.beta_set = utils.get_beta_set()
        self.alpha_set = tf.math.cumprod(1-self.beta_set)

        C1 = self.channels 
        C2 = C1 * 3//2
        C3 = C1 * 2
        style_extractor = nn.StyleExtractor()
        model = nn.DiffusionWriter(num_layers=self.num_attlayers, c1=C1, c2=C2, c3=C3)
        
        _stroke = tf.random.normal([1, 400, 2])
        _text = tf.random.uniform([1, 40], dtype=tf.int32, maxval=50)
        _noise = tf.random.uniform([1, 1])
        _style_vector = tf.random.normal([1, 14, 1280])
        _ = model(_stroke, _text, _noise, _style_vector)
        
        # We have to call the model on input first
        print('[INFO] Model weights loaded ...')
        model.load_weights(self.weights_path)
        self.model = model

        # Create writer style image
        print('[INFO] Style vector created ... ')
        writer_img = tf.expand_dims(preprocessing.read_img(self.writer_source, 96), 0)
        self.style_vector = style_extractor(writer_img)

    def _reset_style_vector(self, writer_source):
        if(not os.path.exists(writer_source)):
            raise Exception("Writer source path do not exist")

        self.writer_source = writer_source
        writer_img = tf.expand_dims(preprocessing.read_img(self.writer_source, 96), 0)
        self.style_vector = style_extractor(writer_img) 
        print('[INFO] Writer source has been reset ...')

    def run_single_generation(self, textstring, output_path, diffmode='new', show=False, seqlen=None):
        timesteps = len(textstring) * 16 if seqlen is None else seqlen
        timesteps = timesteps - (timesteps%8) + 8 

        utils.run_batch_inference(self.model, self.beta_set, textstring, self.style_vector, 
                                    tokenizer=self.tokenizer, time_steps=timesteps, diffusion_mode=diffmode, 
                                    show_samples=show, path=output_path)

    def run_batch_generation(self, textstrings, output_dir, diffmode='new', show=False, seqlen=None):
        if(not os.path.exists(output_dir)):
            print(f'[INFO] Creating output directory {output_dir} ... ')
            os.mkdir(output_dir)

        with tqdm.tqdm(total=len(textstrings)) as pbar:
            for i, textstring in enumerate(textstrings):
                output_path = os.path.join(output_dir, f'sample_{i}')

                self.run_single_generation(textstring, output_path, diffmode=diffmode,
                        show=show, seqlen=seqlen)

                pbar.update(1)

# gen = HandWritingGenerator('weights/model.h5', 'writers/writer-style-01.jpg')
# gen.run_batch_generation(["I hate", "my life"], "data/samples")
