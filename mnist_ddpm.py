import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
import math
from tqdm import tqdm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Diffusion parameters
T = 1000
beta_start = 1e-4
beta_end = 0.02

# Training parameters
batch_size = 128
lr = 1e-3
epochs = 100

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Noise schedule
betas = torch.linspace(beta_start, beta_end, T).to(device)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]])
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.relu(h)

        time_emb = self.time_mlp(t)
        time_emb = time_emb[:, :, None, None]
        h = h + time_emb

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.relu(h)
        return h

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        time_dim = 256
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # Encoder
        self.enc1 = ConvBlock(1, 32, time_dim)
        self.enc2 = ConvBlock(32, 64, time_dim)
        self.enc3 = ConvBlock(64, 128, time_dim)
        self.enc4 = ConvBlock(128, 256, time_dim)

        # Bottleneck
        self.bottleneck = ConvBlock(256, 256, time_dim)

        # Decoder
        self.dec4 = ConvBlock(256, 128, time_dim)
        self.dec3 = ConvBlock(128, 64, time_dim)
        self.dec2 = ConvBlock(64, 32, time_dim)
        self.dec1 = ConvBlock(32, 32, time_dim)

        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x, t):
        t = self.time_mlp(t)

        # Encoder
        x = self.enc1(x, t)
        x = self.enc2(x, t)
        x = self.enc3(x, t)
        x = self.enc4(x, t)

        # Bottleneck
        x = self.bottleneck(x, t)

        # Decoder
        x = self.dec4(x, t)
        x = self.dec3(x, t)
        x = self.dec2(x, t)
        x = self.dec1(x, t)

        return self.final(x)

class SmolMLP(nn.Module):
    def __init__(self):
        super().__init__()
        time_dim = 256
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # MLP layers
        self.fc1 = nn.Linear(28*28 + time_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 28*28)

    def forward(self, x, t):
        batch_size = x.shape[0]
        t = self.time_mlp(t)

        # Flatten image and concatenate with time embedding
        x_flat = x.view(batch_size, -1)
        x = torch.cat([x_flat, t], dim=1)

        # MLP layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        # Reshape back to image
        return x.view(batch_size, 1, 28, 28)

class ChunkyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        time_dim = 256
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # Much larger MLP layers
        self.fc1 = nn.Linear(28*28 + time_dim, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, 2048)
        self.fc5 = nn.Linear(2048, 2048)
        self.fc6 = nn.Linear(2048, 28*28)

    def forward(self, x, t):
        batch_size = x.shape[0]
        t = self.time_mlp(t)

        # Flatten image and concatenate with time embedding
        x_flat = x.view(batch_size, -1)
        x = torch.cat([x_flat, t], dim=1)

        # MLP layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)

        # Reshape back to image
        return x.view(batch_size, 1, 28, 28)

def q_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

def p_sample(model, x, t, t_index):
    betas_t = betas[t].reshape(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    sqrt_recip_alphas_t = sqrt_recip_alphas[t].reshape(-1, 1, 1, 1)

    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = posterior_variance[t].reshape(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample(model, n_samples=25):
    model.eval()
    x = torch.randn(n_samples, 1, 28, 28).to(device)

    for i in tqdm(reversed(range(T)), desc='Sampling', leave=False):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)
        x = p_sample(model, x, t, i)

    model.train()
    return x

def train():
    # Change model type here
    model = ConvNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    os.makedirs('samples', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        total_loss = 0

        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()

            t = torch.randint(0, T, (data.shape[0],), device=device).long()
            noise = torch.randn_like(data)
            x_noisy = q_sample(data, t, noise)
            predicted_noise = model(x_noisy, t)

            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})

        # Sample and save
        samples = sample(model)
        samples = (samples + 1) / 2  # Denormalize
        samples = samples.clamp(0, 1)

        grid = vutils.make_grid(samples, nrow=5, padding=2, normalize=False)
        vutils.save_image(grid, f'samples/epoch_{epoch+1:03d}.png')

        # Checkpoint
        if epoch > 0: # remove previous checkpoint
            prev_checkpoint = f'checkpoints/checkpoint_epoch_{epoch:03d}.pt'
            if os.path.exists(prev_checkpoint):
                os.remove(prev_checkpoint)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'checkpoints/checkpoint_epoch_{epoch+1:03d}.pt')

if __name__ == '__main__':
    for model in [ConvNet(), SmolMLP(), ChunkyMLP()]:
        # ConvNet: 3,276,129
        # SmolMLP: 1,526,288
        # ChunkyMLP: 20,589,584
        print(f"{model.__class__.__name__}: {sum(p.numel() for p in model.parameters()):,}")
    train()
