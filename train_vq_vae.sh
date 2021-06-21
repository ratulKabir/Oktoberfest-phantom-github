# call as: nohup bash train_vq_vae.sh > logs/train_test.log &

# GLOB pattern that identifies all TFRecords of your dataset
tfrecord_pattern=/home/watson/project/oktoberfest/dataset/tfr/output*

# The directory into which your trained model will be saved
save_dir=/project/watson/op_latest_test/team-oktoberfest-phantom/models/testing
#restore_dir=/project/borkar/Oktoberfest/data/tfrecord_orig/codebook_experiment/
# The project dir containing the vq_vae Python module and the vq_vae.gin file
project_dir=/project/watson/op_latest_test/team-oktoberfest-phantom

mkdir $save_dir

# Required, so we can access the vq_vae module
export PYTHONPATH=$project_dir  

ddsp_run \
  --mode=train \
  --alsologtostderr \
  --save_dir="$save_dir" \
  --gin_file="$project_dir/vq_vae.gin" \
  --gin_file="$project_dir/gin_files/tfrecordShuffled.gin" \
  --gin_param="TFRecordProvider_shuffle.file_pattern='$tfrecord_pattern'" \
  --gin_param="batch_size=16" \
  --gin_param="train_util.train.num_steps=50000" \
  --gin_param="train_util.train.steps_per_save=1000" \
  --gin_param="trainers.Trainer.checkpoints_to_keep=20"