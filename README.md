# HAMI-M3D
This repo is the released code of our work **Harmfully Manipulated Images Matter in Multimodal
Misinformation Detection**

Our released code follows to "EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection" and "BDANN: BERT-Based Domain Adaptation Neural
Network for Multi-Modal Fake News Detection"

### Requirements

```
torch==1.12.1
cudatoolkit==11.3.1
transformers==4.27.4
```

### Train

- Prepare the datasets Weibo, Gossip and Twitter. The datasets are from https://github.com/yaqingwang/EANN-KDD18 and https://github.com/shiivangii/SpotFakePlus,
and you should put them in `./Data`

- Run the python file
```shell
cd src
python ./run.py
```

- Check log files in `./log`

### Tips
1. You should manually split the training set of GossipCop into the divisions of training and validation, then, revise the file road in the function `write_data` in line 89, `process_gossipcop.py`
2. We prepare a `auto_logging.py` to automatically read the output log files into an excel table.

### Citation
```

```