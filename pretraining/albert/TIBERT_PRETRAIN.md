# Pretraining TiBERT-base from AlBERT-base-v2

## Preprocess on EC2

First spin an instance up and export the instance ip as a variable on your local machine:

```bash
export INSTANCE_IP=instance_ip
```

### Prepare SSH keys on the instance

Run this on local machine to prep the instance:

```bash
ssh -i ~/credentials/ml-dev-key.pem ubuntu@$INSTANCE_IP mkdir /home/ubuntu/.ssh/
scp -i ~/credentials/ml-dev-key.pem ~/workspace/credentials/aws_github_ssh/* ubuntu@$INSTANCE_IP:/home/ubuntu/.ssh/
ssh -i ~/credentials/ml-dev-key.pem ubuntu@$INSTANCE_IP chmod 700 /home/ubuntu/.ssh/id_rsa
```

To SSH into the instance use this:

```bash
ssh -i ~/credentials/ml-dev-key.pem -o ServerAliveInterval=30 ubuntu@$INSTANCE_IP
```

To sync code onto the instance use this:

```bash
rsync -avL --progress -e "ssh -i ~/credentials/ml-dev-key.pem" ~/workspace/tibert/* ubuntu@$INSTANCE_IP:/home/ubuntu/tibert/
```

Now SSH into the instance.

### Prepare credentials on the instance

```bash
sudo apt-get update
sudo apt-get dist-upgrade awscli -y
aws configure
```

Reboot the instance.

### Clone repos and models

```bash
chmod 700 ~/.ssh/id_rsa
git clone --recurse-submodules git@github.com:eisene/tibert.git
wget https://storage.googleapis.com/albert_models/albert_base_v2.tar.gz
tar -xf albert_base_v2.tar.gz
```

### Install conda and create training environment

First install miniconda.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
eval "$(/home/ubuntu/miniconda/bin/conda shell.bash hook)"
conda init
conda update conda -y
```

Then create the environment.

```bash
cd ~/tibert/tibert/albert_pretrain
conda env create -f tibert_pretrain.yml
```

Now copy the data over.

```bash
mkdir ./training_data
aws s3 cp --recursive s3://eisene-experiments/ulm-training-data/bert-like-endogenous-docs/ ./training_data/
```

Copy over the tokenizers.

```bash
mkdir ./sp_models
aws s3 cp --recursive s3://eisene-experiments/tibert-tokenizers/ ./sp_models/
```

Alternatively you can train a tokenizer from scratch. This is a big job because it uses all the data (probably unnecessary). This also requires you to install sentencepiece from source.

```bash
pip uninstall sentencepiece
```

Now install sentencepiece from source: <https://github.com/google/sentencepiece#c-from-source>. After that you can train.

```bash
chmod 700 ./train_spm_model.sh
cat ./training_data/*.txt > all.txt
train_spm_model.sh ./all.txt 30k-bo-unigram
```

### Run the preprocessing

First create a screen. Don't forget to deactivate your conda environment before creating the screen if you need to.

```bash
conda deactivate
screen -S preprocess_128
conda activate tibert_pretrain
cd ~/tibert/tibert/albert_pretrain
```

Inside the screen now run the preprocessing. This will take a very long time, ~2hours on a c5.9xlarge

```bash
source build_pretraining_data_no_sop.sh ./training_data 128 tibert_spm_bpe_big 89012 1
```

You can vary the random seed and dupe factor if you're running out of memory.

Finally collect the data and upload to S3.

```bash
mkdir training_data/tfrecords-128
mv training_data/*.tf_record training_data/tfrecords-128/
aws s3 cp --recursive ./training_data/tfrecords-128 s3://eisene-experiments/ulm-training-data/bert-like-endogenous-docs-tfrecords-128
```

## Training on a GCP TPU

### Prepare the VM with SSH keys

Run this on your local machine.

```bash
gcloud compute scp ~/workspace/credentials/aws_github_ssh/* preprocess-vm:/home/eeisenst/.ssh
```

Then SSH into the VM.

```bash
gcloud compute ssh preprocess-vm
```

### Clone repos and models

First clone the TiBERT repo.

```bash
chmod 700 ~/.ssh/id_rsa
git clone --recurse-submodules git@github.com:eisene/tibert.git
```

Then download the AlBERT-base-v2 model, you will need the config file.

```bash
wget https://storage.googleapis.com/albert_models/albert_base_v2.tar.gz
tar -xf albert_base_v2.tar.gz
```

If you want to train without the SOP task _from scratch_, make a new AlBERT config file.

```bash
vim ~/albert_base/albert_config_no_sop.json
```

The contents that you need to change are:

```json
{
  "type_vocab_size": 1,
}
```

If you're restarting training for a model that was pre-trained with SOP, don't use a new config file.

### Create the environment

```bash
cd ~/tibert/tibert/albert_pretrain
conda env create -f tibert_pretrain.yml
conda activate tibert_pretrain
```

### Test the setup

This runs training with batch size 512 on CPU, it should OOM:

```bash
python -m albert.run_pretraining_no_sop \
    --input_file=gs://eeisenst-experiments/bert-like-endogenous-docs-tfrecords-128/*.tf_record \
    --output_dir=gs://eeisenst-experiments/training-results/tibert-base-128-no-sop/ \
    --albert_config_file=/home/eeisenst/albert_base/albert_config.json \
    --do_train \
    --train_batch_size=512 \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --optimizer=lamb \
    --learning_rate=.00176 \
    --num_train_steps=1000000 \
    --num_warmup_steps=3125 \
    --save_checkpoints_steps=5000
```

### Create the TPU

Your instance will need full API permissions for this. This creates a preemptible TPU.

```bash
gcloud compute tpus create tibert-tpu \
    --zone=us-central1-a \
    --network=default \
    --version=1.15.2 \
    --accelerator-type=v3-8 \
    --preemptible
```

Sometimes there may be contention, especially on the v3-8 TPUs. Try a v2-8 in this case.

```bash
gcloud compute tpus create tibert-tpu \
    --zone=us-central1-b \
    --network=default \
    --version=1.15.2 \
    --accelerator-type=v2-8 \
    --preemptible
```

Note the different zone, you will need to change the corresponding parameter in the training script arguments.

### Run the training

First create a screen. Don't forget to deactivate your conda environment before creating the screen if you need to.

```bash
conda deactivate
screen -S training
conda activate tibert_pretrain
cd ~/tibert/tibert/albert_pretrain
```

Inside the screen now run the training.

```bash
python -m albert.run_pretraining_no_sop \
    --input_file=gs://eeisenst-experiments/bert-like-endogenous-docs-tfrecords-128/*.tf_record \
    --output_dir=gs://eeisenst-experiments/training-results/tibert-base-128-no-sop/ \
    --albert_config_file=/home/eeisenst/albert_base/albert_config.json \
    --do_train \
    --train_batch_size=1024 \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --optimizer=lamb \
    --learning_rate=.00176 \
    --num_warmup_steps=3125 \
    --num_train_steps=1000000 \
    --save_checkpoints_steps=1000 \
    --keep_checkpoint_max=100 \
    --use_tpu=true \
    --tpu_name=tibert-tpu \
    --tpu_zone=us-central1-a \
    --num_tpu_cores=8
```

Modify as needed. The lowest learning rate I used with 1024 batch size was 0.00022.

You can SSH directly into the training screen from your local machine.

```bash
gcloud compute ssh albert-vm --command="screen -R training" -- -t
```

## Validate a training run

To validate the training run on your local machine first create the folders where you will put the run results, then copy over the model checkpoint.

```bash
gsutil ls gs://eeisenst-experiments/training-results/tibert-base-128-no-sop/*
gsutil -m cp gs://eeisenst-experiments/training-results/tibert-base-128-no-sop/model.ckpt-10000* c:/workspace/tibert_data/test-data/test-model/
```

Clean up any PyTorch converted checkpoint currently there.

```bash
rm c:/workspace/tibert_data/test-data/test-model/pytorch_model.bin
```

Then activate the pretraining environment.

```bash
conda activate tibert_pretrain
```

Finally convert the TF checkpoint to Transformers PyTorch.

```bash
transformers-cli convert --model_type albert \
    --tf_checkpoint c:/workspace/tibert_data/test-data/test-model/model.ckpt-10000 \
    --config c:/workspace/tibert_data/training/models/albert_base/albert_config_no_sop.json \
    --pytorch_dump_output c:/workspace/tibert_data/test-data/test-model/pytorch_model.bin
```

You can now open this in Jupyter using the "Training Run Testing" notebook and reproduce the training cross-entropy using Transformers.
