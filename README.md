# Diffusion Models for Hand-writing generation
Tensorflow implementation for [Diffusion Models for Hand-Writing Generation](https://arxiv.org/abs/2011.06704)

### 1. Download dataset
First, download and extract the contents of the following files:
 - lineStrokes-all.tar.gz (Pen strokes in XML)
 - lineImages-all.tar.gz (Images for style encoding)
 - ascii-all.tar.gz (Sequences of text)
<br/>

from [this dataset URL](https://fki.tic.heia-fr.ch/databases/download-the-iam-on-line-handwriting-database) and extract the
contents to ./data

### 2. Training 
To start training, log in to wandb.ai account on the CLI
Then, run the train.py script with the following arguments :
```
python3 train.py --steps <number_of_epochs>
					--batchsize <batch_size>
					--seqlen <max_text_sequence_length>
					--textlen <max_text_length>
					--width <style_img_width>
					--warmup <lr_scheduler_warmup_step>
					--dropout <dropout_rate>
					--num_attlayers <number_attention_layers>
					--print_every <show_loss_every_m_steps>
					--save_every <ckpt_weights_every_n_steps>
```

### 3. Inference
To run sample inference, run the inference.py :
```
python3 inference.py --textstring <text_you_want>
						--writersource <style_image>
						--name <output_file_name>
						--weights <path_to_weights.h5>
```
