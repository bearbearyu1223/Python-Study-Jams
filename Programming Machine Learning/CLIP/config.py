import torch
debug = True
debug_sample_size = 8000
image_path = "Datasets/Images"
test_dataset_path = "Datasets/Test"
test_dataset_filename = "test.csv"
valid_dataset_path = "Datasets/Validate"
valid_dataset_filename = "validate.csv"
captions_path = "Datasets"
batch_size = 32
num_workers = 2
lr = 1e-3
weight_decay = 1e-5
patience = 1
factor = 0.8
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'resnet50'
image_embedding = 2048
text_encoder_model = "distilbert-base-uncased"
text_embedding = 768
text_tokenizer = "distilbert-base-uncased"
max_length = 200

pretrained = True # for both image encoder and text encoder
trainable = True # for both image encoder and text encoder
temperature = 1.2

# image size
size = 224

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256
dropout = 0.1

# split ratio of train, validation, and test dataset
test_ratio = 0.1
train_valid_split = 0.8
