### Smart Composer (QuickStart)
1. Clone this Repo to your local directory 
2. Create a Conda Env using the `env.yml` file
```commandline
cd Smart\ Compose
conda env create -f env.yml
```
3. Download the pre-trained GloVe Embeddings and save the embedding files under the `data` directory
```commandline
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip -q glove.6B.zip
```
4. Run trainer.py to see training and prediction on a small sample dataset.
```commandline
python trainer.py
```
sample results will be printed out in console:
```shell
Model: "encoder_decoder_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
encoder (Encoder)            multiple                  177348    
_________________________________________________________________
decoder (Decoder)            multiple                  271901    
=================================================================
Total params: 449,249
Trainable params: 328,793
Non-trainable params: 120,456
_________________________________________________________________
Train on 711 samples, validate on 178 samples
... ...
Epoch 24/50
704/711 [============================>.] - ETA: 0s - loss: 0.0262 - sparse_categorical_crossentropy: 0.0663
711/711 [==============================] - 7s 10ms/sample - loss: 0.0260 - sparse_categorical_crossentropy: 0.0660 - val_loss: 0.3797 - val_sparse_categorical_crossentropy: 0.9186

Epoch 00024: early stopping
'input seq : i have'
'predict seq : been involved in'
'input seq : we can'
'predict seq : get back to'
'input seq : please mark'
'predict seq : you want to'
'input seq : follow these'
'predict seq : steps so you'


```
