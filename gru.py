import os
import tensorflow as tf
import argparse
import numpy as np
import IPython.display as ipd
from IPython.display import Audio
import time
import matplotlib.pyplot as plt
# from utils.util_funcs import generate_audio, generate_sample_test

tfkl = tf.keras.layers
parser = argparse.ArgumentParser()
parser.add_argument('--folder_name', '--folder_name', help='name of the folder. The model and the tensorboard will be stored in their respective location with this same folder name.', required=True)
# parser.add_argument('')
args = parser.parse_args()
folder_name = args.folder_name

#load latent space
batch_size = 32

latent_files_path = 'saved_latent_spaces/ddsp_shuffled_32_128_codebook_usuage-latent_space'

training_code_inds = np.load('{}/train_code_inds.npy'.format(latent_files_path))
training_codes = np.load('{}/train_codes.npy'.format(latent_files_path))
codebook = np.load('{}/codebook.npy'.format(latent_files_path))

code_data = tf.data.Dataset.from_tensor_slices((training_code_inds, training_codes))
# cache the dataset to memory to get a speedup while reading from it.
code_data = code_data.cache()
code_data_ready = code_data.shuffle(50000).batch(batch_size, drop_remainder=True)#.repeat()

seqlen = 1000
dim_code = codebook.shape[-1] + 1 + 1

inputs = tfkl.Input(batch_shape=(batch_size, None, dim_code))
x = tfkl.GRU(512, return_sequences=True, stateful=True)(inputs)
x = tfkl.GRU(512, return_sequences=True, stateful=True)(x)
x = tfkl.GRU(512, return_sequences=True, stateful=True)(x)

f0_output = tfkl.Dense(1998)(x)
ld_output = tfkl.Dense(121)(x)
z_output = tfkl.Dense(codebook.shape[0])(x)
    
model_rnn = tf.keras.Model(inputs=inputs, outputs=[f0_output, ld_output, z_output], name='Functional-api-RNN')

EPOCHS = 200
# train_steps = 200000
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
lr = tf.optimizers.schedules.PolynomialDecay(0.001, EPOCHS*1200, 0.000001)
opt = tf.optimizers.Adam(lr)

model_rnn.summary()

checkpoint_path = "models/ratul/{}".format(folder_name)
summary_dir = "models/ratul/tensorboard/{}".format(folder_name)

ckpt = tf.train.Checkpoint(model=model_rnn,
                           optimizer=opt)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')
    
@tf.function
def train(ind_batch, code_batch):
    targets_f0 = ind_batch[:, 1:, 0:1] 
    targets_f0 = tf.reshape(targets_f0, [-1,seqlen-1])
    targets_ld = ind_batch[:, 1:, 1:2] 
    targets_ld = tf.reshape(targets_ld, [-1,seqlen-1]) * (-1)
    targets_z = ind_batch[:, 1:, 2:]
    targets_z = tf.reshape(targets_z, [-1,seqlen-1])
    inp = code_batch[:, :-1]
    
    scale_z_loss = 2
    scale_f0_loss = 1
    scale_ld_loss = 3
    with tf.GradientTape() as tape:
        out = model_rnn(inp)
        xent_f0 = loss(targets_f0, out[0])
        xent_ld = loss(targets_ld, out[1])
        xent_z = loss(targets_z, out[2])
        xent = scale_f0_loss * xent_f0 + scale_ld_loss * xent_ld + scale_z_loss * xent_z
    grads = tape.gradient(xent, model_rnn.trainable_variables)
    opt.apply_gradients(zip(grads, model_rnn.trainable_variables))

    return xent_f0, xent_ld, xent_z, xent, out

# losses = []
summary_writer = tf.summary.create_file_writer(summary_dir) 

with summary_writer.as_default():
    print('Training is starting ...')
    for epoch in range(EPOCHS):
        start = time.time()

        for batch, (inds, codes) in enumerate(code_data_ready):
            model_rnn.reset_states()
            xent_f0, xent_ld, xent_z, xent, out = train(inds, codes)
    #         losses.append(xent)
            if batch % 200 == 0 and batch != 0:
                print ('Epoch {} Batch {} z_loss: {:.4f}, f0_loss: {:.4f}, ld_loss: {:.4f}, total_loss: {:.4f}'.format(
                     epoch + 1, batch, xent_z, xent_f0, xent_ld, xent))
                tf.summary.scalar('Individual losses/Z loss', xent_z, step=batch+(epoch*1200))
                tf.summary.scalar('Individual losses/F0 loss', xent_f0, step=batch+(epoch*1200))
                tf.summary.scalar('Individual losses/Loudness loss', xent_ld, step=batch+(epoch*1200))
                tf.summary.scalar('Total loss', xent, step=batch+(epoch*1200))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                 ckpt_save_path))

        print ('Epoch {} Loss {:.4f}'.format(epoch + 1, xent))
#         gen = generate_sample_test(model_rnn, codebook, codes, chunk_len=1, seqlen=1, batch_size=batch_size)
#         gen_random = generate_audio(model_rnn, codebook, seqlen, batch_size=batch_size)
#         ipd.display(Audio(gen_random[0],rate=16000))
#         ipd.display(Audio(gen[0],rate=16000))
        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
