import utils
import tensorflow as tf

MAX_SEQ_LEN = 480  
MAX_TEXT_LEN = 50 
WIDTH =  1400

beta_set = utils.get_beta_set()
alpha_set = tf.math.cumprod(1-beta_set)
# print(beta_set)
# print(alpha_set)

path = './data/train_strokes.p'
strokes, texts, samples = utils.preprocess_data(path, MAX_TEXT_LEN, MAX_SEQ_LEN, WIDTH, 96)
alphas = utils.get_alphas(len(strokes), alpha_set)

# print(strokes[0].shape, texts.shape, samples.shape)
# print(texts)
# print(alphas, alphas.shape)




