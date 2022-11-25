import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
import os
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

# ROOT_DIR = "/home/sean/GAN/pixel2pixel/archive/maps/maps/train"
def train_fn(disc,gen,loader,opt_disc,opt_gen,l1_Loss,bce,g_scalar,d_scalar):
    loop = tqdm(loader,leave=True)

    for idx,(x,y)in enumerate(loop):
        x,y = x.to(config.DEVICE),y.to(config.DEVICE)

        #tain discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x,y)
            D_fake = disc (x,y_fake.detach())
            
            D_real_loss = bce(D_real,torch.ones_like(D_real))
            D_fake_loss = bce(D_fake,torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2 

        opt_disc.zero_grad()
        d_scalar.scale(D_loss).backward()
        d_scalar.step(opt_disc)
        d_scalar.update()


        #train Generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x,y_fake)
            G_fake_loss = bce(D_fake,torch.ones_like(D_fake))
            L1 = l1_Loss(y_fake,y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scalar.scale(G_loss).backward()
        g_scalar.step(opt_gen)
        g_scalar.update()



def main():
    if os.path.exists(config.SAMPLE_DIR) == False:
        os.mkdir(config.SAMPLE_DIR)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(),lr = config.LEARNING_RATE,betas=(0.5,0.999))
    opt_gen = optim.Adam(gen.parameters(),lr = config.LEARNING_RATE,betas=(0.5,0.999))
    BCE = nn.BCEWithLogitsLoss()#听说wgan的loss和patch gan 的结合效果不好
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN,gen,opt_gen,config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC,disc,opt_disc,config.LEARNING_RATE)

    train_dataset = MapDataset(root_dir = config.TRAIN_DIR)
    train_loader = DataLoader(train_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=config.NUM_WORKERS)
    g_scaler = torch.cuda.amp.GradScaler()
    disc_scalar = torch.cuda.amp.GradScaler()
    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range (config.NUM_EPOCHS):
        train_fn(disc,gen,train_loader,opt_disc,opt_gen,L1_LOSS,BCE,g_scaler,disc_scalar)
        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen,opt_gen,filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc,opt_disc,filename=config.CHECKPOINT_DISC)

        save_some_examples(gen,val_loader,epoch,folder=config.SAMPLE_DIR)



main()