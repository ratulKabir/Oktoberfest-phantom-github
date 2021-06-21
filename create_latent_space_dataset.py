import os
import ddsp
import vq_vae
import tensorflow as tf
import gin
import numpy as np

gin.parse_config_file('models/ratul/ddsp_shuffled_32_128_codebook_usuage/operative_config-0.gin')
checkpoint_dir = "models/ratul/ddsp_shuffled_32_128_codebook_usuage/"

# VQ-VAE model
model_vq = vq_vae.QuantizingAutoencoder()
model_vq.restore(checkpoint_dir)

data_provider = ddsp.training.data.TFRecordProvider('../data/tfr/all/*')

dataset = data_provider.get_dataset()#.shuffle(50000)

"""Create training set of codes"""

train_steps = 50000

f0_hzs = []
f0_scaleds = []
ld_scaleds = []
zs = []
loudness_dbs = []
train_code_inds = []
c = 0
for data in dataset:        
    data['audio'] = data['audio'][None,:]   
    conditioning = model_vq.encode(data, training=False)
    conditioning = model_vq.quantizer(conditioning)
    
    loudness_dbs.append(conditioning['loudness_db'])
    f0_hzs.append(conditioning['f0_hz'])
    f0_scaleds.append(conditioning['f0_scaled'])
    ld_scaleds.append(conditioning['ld_scaled'])
#     zs.append(conditioning['z'])
#     train_code_inds.append(conditioning['z_indices'])
    print(c,end='\r')
    c += 1
    
# train_code_inds = np.concatenate(train_code_inds)
f0_scaleds = np.concatenate(f0_scaleds)
ld_scaleds = np.concatenate(ld_scaleds)
loudness_dbs = np.concatenate(loudness_dbs)
f0_hzs = np.concatenate(f0_hzs)
# train_codes = np.concatenate(zs)

training_codes = np.concatenate((f0_scaleds, ld_scaleds, train_codes), axis=-1)

training_code_inds = np.reshape(train_code_inds, [int(train_code_inds.shape[0]/1000), 1000, 1])
training_code_inds = np.concatenate((tf.cast(f0_hzs, tf.dtypes.int64), 
                                     tf.cast(loudness_dbs, tf.dtypes.int64), 
                                     training_code_inds), axis=-1)
codebook = model_vq.quantizer.codebook

latent_files_path = 'saved_latent_spaces/ddsp_shuffled_32_128_codebook_usuage-latent_space-talk_orig'
if not os.path.exists(latent_files_path):
    os.mkdir(latent_files_path)

np.save('{}/train_code_inds.npy'.format(latent_files_path), training_code_inds)
np.save('{}/train_codes.npy'.format(latent_files_path), training_codes)
np.save('{}/codebook.npy'.format(latent_files_path), codebook.numpy())