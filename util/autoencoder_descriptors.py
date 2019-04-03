from train_cnn_autoencoder import autoencoder
import torch

def generate_autoencoder_desc(img_tensor, model_parameters_path):
    ##### img_tensor shape [128, 3, 64, 64]
    model = autoencoder(model_parameters_path, pretrained=True).cuda()
    model.eval()
    encoder, decoder = model(img_tensor.cuda())
    encoder = encoder.cpu().detach().numpy()
    return encoder



    

