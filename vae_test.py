import torch
from torchvision import transforms
from PIL import Image
from vae_train import VAE
import matplotlib.pyplot as plt
import numpy as np

image_path = './outputs/UE4 and Airsim/20240417-155906/results/Depth/image_19937.png'
model_path = './vae.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_vae(model_path, image_path):
    model = VAE().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((144, 256)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert('L')
    img = transform(img).unsqueeze(0).to(device)
    print(img.shape)
    print(img)

    with torch.no_grad():
        recon, _, _ = model(img)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img[0].cpu().detach().numpy().squeeze(0), cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(recon[0].cpu().detach().numpy().squeeze(0), cmap='gray')
    ax[1].set_title('Reconstructed Image')
    ax[1].axis('off')
    plt.show()

if __name__ == '__main__':
    print('Start testing')
    test_vae(model_path, image_path)
