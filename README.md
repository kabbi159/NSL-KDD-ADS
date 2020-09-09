# NSL-KDD-ADS
Unsupervised Anomaly Detection System PyTorch Implementation of NSL-KDD Dataset

## Data Preprocessing
Download the dataset from https://www.unb.ca/cic/datasets/nsl.html   
I followed the preprocessing by @kyuyeonpoor's [nsl-kdd-autoencoder](https://github.com/kyuyeonpooh/nsl-kdd-autoencoder). (This setting can be modified)

## Reconsturction Based Models
* AE (Autoencoder)

* DSEBM (Deep Structed Energy Based Models for Anomaly Detection)
  * https://arxiv.org/abs/1605.07717 (ICML 2016)
 
* MEMAE (Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection)
  * https://arxiv.org/abs/1904.02639 (ICCV 2019)   
  I followed the experimental details and hyperparmeters from the paper in **4.3 Experiments on Cybersecurity Data** except input shape of the encoder and output of the decoder. (It depends on the data preprocessing)   
  Also, I referenced author's code ([3d-conv MEMAE](https://github.com/donggong1/memae-anomaly-detection)). This implementation is about fc-MEMAE.
  

## To-Do
- [ ] Implement DSEBM
- [ ] Solve 'set_seed' problem in MEMAE
