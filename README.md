# TTS Project to train DeepSpeech2

## Installation guide

You first run this in your shell:
```shell
git clone https://github.com/mizoru/FastSpeech2.git
cd FastSpeech2
pip install -r requirements.txt
wget https://dagshub.com/mizoru/FastSpeech2/raw/be43b4f7d3be88e258e0cef8cdd68d587fff54e7/checkpoint1.pth
```

## Inference

To get the audio for sentences in texts.txt run: 
```shell
python test.py -m checkpoint1.pth -t texts.txt
```

## Fine-Tuning

To continue training:
```shell
python train.py -r checkpoint1.pth
```

## Results

You can find the final inference-time predictions in wavresults.zip

## Training Logs
You can find the report on the training (here)[https://wandb.ai/mizoru/FastSpeech2/reports/Training-FastSpeech2--VmlldzozMDUxMDA1] and also look at my (WandB logs)[https://wandb.ai/mizoru/FastSpeech2]
