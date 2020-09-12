# NSL-KDD-ADS
Unsupervised Anomaly Detection System PyTorch Implementation of NSL-KDD Dataset

## Data Preprocessing
Download the dataset from https://www.unb.ca/cic/datasets/nsl.html   
I followed the preprocessing by @kyuyeonpoor's [nsl-kdd-autoencoder](https://github.com/kyuyeonpooh/nsl-kdd-autoencoder). (This setting can be modified)

## Autoencoder Based Models
* AE (Autoencoder)
* DSEBM (Deep Structed Energy Based Model)
* DAGMM
* MEMAE (Memory-augmented Deep Autoencoder)

### DSEBM (Deep Structed Energy Based Models for Anomaly Detection) [[paper](https://arxiv.org/abs/1605.07717)] - ICML 2016   
To be implemented

### DAGMM (Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection) [[paper](https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf)] - ICLR 2018
To be implemented

### MEMAE (Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection) [[paper](https://arxiv.org/abs/1904.02639)] - ICCV 2019    
I followed the experimental details and hyperparmeters from the paper in **4.3 Experiments on Cybersecurity Data** except input shape of the encoder and output of the decoder. (It depends on the data preprocessing)   
Also, I referenced author's code ([3d-conv MEMAE](https://github.com/donggong1/memae-anomaly-detection)). This implementation is about fc-MEMAE.   
```bash
python main_memae.py
```

## To-Do
- [ ] Implement DSEBM and DAGMM
- [ ] set seed implementation
- [ ] hyperparmeter setting by argument
