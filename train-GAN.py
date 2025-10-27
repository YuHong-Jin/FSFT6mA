import os
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

GDIM = 512
DDIM = 86
FIXED_GENERATOR = False
LAMBDA = 0.1
CRITIC_ITERS = 7
ITERS = 1000
use_cuda = True

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(2720, GDIM),
            nn.ReLU(True),
            nn.Linear(GDIM, GDIM),
            nn.ReLU(True),
            nn.Linear(GDIM, GDIM),
            nn.Tanh(),
            nn.Linear(GDIM, 2720)
        )

    def forward(self, noise, real_data):
        if FIXED_GENERATOR:
            return noise + real_data
        else:
            return self.main(noise)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(2720, DDIM)
        self.fc2 = nn.Linear(DDIM, DDIM)
        self.fc3 = nn.Linear(DDIM, DDIM)
        self.fc4 = nn.Linear(DDIM, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        out = self.fc4(h3)
        return out.view(-1), h1, h2, h3


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)


def calc_gradient_penalty(netD, real_data, fake_data, batch_size):
    alpha = torch.rand(batch_size, 1).expand(real_data.size())
    if use_cuda:
        alpha = alpha.cuda()
    interpolates = autograd.Variable(alpha * real_data + (1 - alpha) * fake_data, requires_grad=True)
    disc_interpolates, _, _, _ = netD(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates).cuda() if use_cuda else torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA


def load_dataset(file_path):
    with open(file_path) as f:
        matrix = [list(map(float, x.strip().split(","))) for x in f]
    features = [line[:2720] for line in matrix]
    return np.array(features, dtype=np.float32)


def train_gan(dataset, iters=ITERS, batch_size=None):
    if batch_size is None:
        batch_size = len(dataset)

    netG = Generator()
    netD = Discriminator()
    netG.apply(weights_init)
    netD.apply(weights_init)

    if use_cuda:
        netG.cuda()
        netD.cuda()

    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.99))
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.99))

    one = torch.tensor(1., dtype=torch.float).cuda() if use_cuda else torch.tensor(1., dtype=torch.float)
    mone = -one

    for iteration in range(iters):
        for p in netD.parameters():
            p.requires_grad = True

        real_data = torch.FloatTensor(dataset)
        if use_cuda:
            real_data = real_data.cuda()
        real_data_v = autograd.Variable(real_data)

        for _ in range(CRITIC_ITERS):
            netD.zero_grad()
            D_real, _, _, _ = netD(real_data_v)
            D_real_mean = D_real.mean()
            D_real_mean.backward(mone)

            noise = torch.randn(batch_size, 2720)
            if use_cuda:
                noise = noise.cuda()
            fake = autograd.Variable(netG(noise, real_data_v).data)
            D_fake, _, _, _ = netD(fake)
            D_fake_mean = D_fake.mean()
            D_fake_mean.backward(one)

            gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data, batch_size)
            gradient_penalty.backward()
            optimizerD.step()

        if iteration % 200 == 0:
            fake_output = fake.data.cpu().numpy()
            output_file = f"05-results/Ath{iteration}_Synthetic_Training1.txt"
            np.savetxt(output_file, fake_output, delimiter=",")

        if not FIXED_GENERATOR:
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()
            noise = torch.randn(batch_size, 2720)
            if use_cuda:
                noise = noise.cuda()
            fake = netG(noise, real_data_v)
            G, _, _, _ = netD(fake)
            G_mean = G.mean()
            G_mean.backward(mone)
            optimizerG.step()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset = load_dataset("A.thaliana_features_label1.txt")
    train_gan(dataset)