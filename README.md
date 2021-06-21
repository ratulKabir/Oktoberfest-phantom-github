# DDSP VQ-VAE
This is a 6 credits team project done by me and two of my fellow students.

# Getting started
- Install the project's requirements using `pip install -r requirements.txt`
	- We have used ddsp==0.5.1 but I just saw that version 0.10.0 has been published. Our code might work on that newer version but it is also possible that it will not be compatible.
	- Make sure to use the correct numba version because a too new version caused trouble for us. Might have been fixed in 0.10.0.

**#Latest update:**
- Use the virtual environment - **workdir**(/project/oktoberfest/) for the environment setup using the latest librares for the project.
- command: source workdir/bin/activate
- To configure the virtual environment with jupyter notebooks: python3 -m ipykernel install --user --name==workdir
- Prepare your Dataset as TFRecords: https://github.com/magenta/ddsp/tree/master/ddsp/training/data_preparation
- Add in all you local paths in the `train_vq_vae.sh` script
- Execute the `train_vq_vae.sh` script


## Generating audio using Autoregressive model:

- A dataset must be created using the latent variabls and the loudness and the fundamental frequency values. Dataset must be saved in a folder named `saved_latent_spaces/`. 
- The following command will create the dataset using a trained DDSP model and save it in the necessary location. <br>
`$python3 create_latent_space_dataset.py`
- Then running the `gru.py` file using the command below will run the autoregresive model on the above dataset. <br>
  `$python3 gru.py --folder_name=tensorboard_folder` <br>
   The tensorboard information will be stored in a folder named `tensorboard_folder`
- Finally, the functions in the `utils` folder can be used to generate audio using the trained model.
