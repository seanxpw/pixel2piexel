"""PatchGan 到底是多少*多少的还存疑，原文使用的是70*70"""
"""CNNBlock 里面之前没有padding = 1 """
import torch

def mask_init(remaining_rate, img_size, device = "cpu"):
    mask = torch.rand([img_size,img_size]).to(device)
    mask = torch.div(mask - (0.5 - remaining_rate),0.5,rounding_mode='trunc')
    mask = mask.repeat(3,1,1)

    return mask

import torch
import torch.nn as nn

# ====================判别器========================= #
class CNNBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=2) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size=4,stride=stride,bias=False,padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self,x):
        return self.conv(x)

#x,y <- concatennate these along the channel
class Discriminator(nn.Module):
    def __init__(self,in_channels = 3,features = [64,128,256,512]) -> None:
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2,features[0],kernel_size=4,stride=2,padding=2,padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        # (256-4+2*1)/2 + 1  = 128
        # 6->64

        layers = []
        in_channels = features[0] #64
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels,feature,stride=1 if feature == features[-1] else 2),
                # 64->128 128->63
                #128->256 63->30
                #256->512 30->27
            )
            in_channels = feature

        
        layers.append(
            nn.Conv2d(
                in_channels,1,kernel_size=4,stride=1,padding=1,padding_mode="reflect"
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self,x,y):
        x= torch.cat([x,y],dim=1)
        x= self.initial(x)
        return self.model(x)

# def test():
#     x = torch.randn((1,3,256,256))
#     y = torch.randn((1,3,256,256))
#     model = Discriminator()
#     preds = model(x,y)
#     print(preds.shape)

# ===========================生成器==================
import torch
import torch.nn as nn
import torch.nn.functional as F


##基本上直接搬的别人的partialConv实现##
class PartialConvLayer (nn.Module):

	def __init__(self, in_channels, out_channels, bn=True, bias=False, sample="none-3", activation="relu"):
		super().__init__()
		self.bn = bn

		if sample == "down-7":
			# Kernel Size = 7, Stride = 2, Padding = 3
			self.input_conv = nn.Conv2d(in_channels, out_channels, 7, 2, 3, bias=bias)
			self.mask_conv = nn.Conv2d(in_channels, out_channels, 7, 2, 3, bias=False)

		elif sample == "down-5":
			self.input_conv = nn.Conv2d(in_channels, out_channels, 5, 2, 2, bias=bias)
			self.mask_conv = nn.Conv2d(in_channels, out_channels, 5, 2, 2, bias=False)

		elif sample == "down-3":
			self.input_conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=bias)
			self.mask_conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False)

		else:
			self.input_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
			self.mask_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)

		nn.init.constant_(self.mask_conv.weight, 1.0)#初始化为1.0, "true"
		# 因为初始化的时候都设置成1了，那么最后卷积的时候就会变成sum(M)

		# "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
		# negative slope of leaky_relu set to 0, same as relu
		# "fan_in" preserved variance from forward pass
		nn.init.kaiming_normal_(self.input_conv.weight, a=0, mode="fan_in")#正态分布

		#mask不需要梯度
		for param in self.mask_conv.parameters():
			param.requires_grad = False

		if bn:
			# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
			# Applying BatchNorm2d layer after Conv will remove the channel mean
			self.batch_normalization = nn.BatchNorm2d(out_channels)

		if activation == "relu":
			# Used between all encoding layers
			self.activation = nn.ReLU()
		elif activation == "leaky_relu":
			# Used between all decoding layers (Leaky RELU with alpha = 0.2)
			self.activation = nn.LeakyReLU(negative_slope=0.2)

	def forward(self, input_x, mask):
		# output = W^T dot (X .* M) + b
		#输入的特征只有没有挖空的部分，，
		output = self.input_conv(input_x * mask)

		# requires_grad = False
		with torch.no_grad():
			# mask = (1 dot M) + 0 = M
			output_mask = self.mask_conv(mask)
        
        # 这个bias是想干嘛捏？？
		if self.input_conv.bias is not None:
			# spreads existing bias values out along 2nd dimension (channels) and then expands to output size
			output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
		else:
			output_bias = torch.zeros_like(output)

		# mask_sum is the sum of the binary mask at every partial convolution location
		mask_is_zero = (output_mask == 0) # 为0的地方就变成True了
		# temporarily sets zero values to one to ease output calculation 
		mask_sum = output_mask.masked_fill_(mask_is_zero, 1.0)

		#这里先将mask是0的地方置为1
		#然后剩下的地方就是sum（M）
        #解释一下，就是之前的mask经过卷积的值会变成一个区域内的和
		#最后做除法的话就自然达到了sum(1)/sum(M)

		# output at each location as follows:
		# output = (W^T dot (X .* M) + b - b) / M_sum + b ; if M_sum > 0
		# output = 0 ; if M_sum == 0
		output = (output - output_bias) / mask_sum + output_bias
		output = output.masked_fill_(mask_is_zero, 0.0)#之前mask是0的地方

		# mask is updated at each location
		new_mask = torch.ones_like(output)
		new_mask = new_mask.masked_fill_(mask_is_zero, 0.0)


		if self.bn:
			output = self.batch_normalization(output)

		if hasattr(self, 'activation'):
			output = self.activation(output)

		return output, new_mask

class Generator(nn.Module):
    def __init__(self,in_channels = 3,features = 64) -> None:
        super().__init__()
        layers = 8

        '''
        那么就先完全按照NV那片论文的结构实现网络好了
        8下8上
        down7和up1拼接
        down8 = BOTTEL_NECK
        down1和up7拼接
        input和up8拼接
        '''
        # “down” for encoder and "up" for decoder
        # ======================= ENCODING LAYERS =======================
        '''注释是不太对的，原文是512*512的输入，所以说我也准备输入512*512的
        如果输入256的话，可以减少一层，不过应该不管也没有关系
        有空看看'''
		# 3x256x256 --> 64x128x128

        self.encoder_1 = PartialConvLayer(3, 64, bn=False, sample="down-7")
		# 64x128x128 --> 128x64x64
        self.encoder_2 = PartialConvLayer(64, 128, sample="down-5")

		# 128x64x64 --> 256x32x32
        self.encoder_3 = PartialConvLayer(128, 256, sample="down-5")

		# 256x32x32 --> 512x16x16
        self.encoder_4 = PartialConvLayer(256, 512, sample="down-3")
        # 512x16x16 --> 512x8x8 --> 512x4x4 --> 512x2x2
        for i in range(5, layers + 1):
            name = "encoder_{:d}".format(i)
            setattr(self, name, PartialConvLayer(512, 512, sample="down-3"))

		# ======================= DECODING LAYERS =======================

        # 这里初始化1到4层的decoder
        for i in range(1, 4+1):
            name = "decoder_{:d}".format(i)
            setattr(self, name, PartialConvLayer(512 + 512, 512, activation="leaky_relu"))

		# UP(512x16x16) + 256x32x32(enc_3 output) = 768x32x32 --> 256x32x32
        self.decoder_5 = PartialConvLayer(512 + 256, 256, activation="leaky_relu")

		# UP(256x32x32) + 128x64x64(enc_2 output) = 384x64x64 --> 128x64x64
        self.decoder_6 = PartialConvLayer(256 + 128, 128, activation="leaky_relu")

		# UP(128x64x64) + 64x128x128(enc_1 output) = 192x128x128 --> 64x128x128
        self.decoder_7 = PartialConvLayer(128 + 64, 64, activation="leaky_relu")

		# UP(64x128x128) + 3x256x256(original image) = 67x256x256 --> 3x256x256(final output)
        self.decoder_8 = PartialConvLayer(64 + 3, 3, bn=False, activation="", bias=True)

    def forward(self,input,mask):
        ## ============== ENCODING START =============== ##
        down1,down_mask1 = self.encoder_1(input,mask)
        down2,down_mask2 = self.encoder_2(down1,down_mask1)
        down3,down_mask3 = self.encoder_3(down2,down_mask2)
        down4,down_mask4 = self.encoder_4(down3,down_mask3)
        down5,down_mask5 = self.encoder_5(down4,down_mask4)
        down6,down_mask6 = self.encoder_6(down5,down_mask5)
        down7,down_mask7 = self.encoder_7(down6,down_mask6)
        bottel_neck,bottel_neck_mask = self.encoder_8(down7,down_mask7)

        ## ============== ENCODING END =============== ##
        ## ============== DECODING START =============== ##应该有更好的写法

        up1 =  F.interpolate(bottel_neck,scale_factor=2)
        up_mask1 =  F.interpolate(bottel_neck_mask,scale_factor=2)
        up1,up_mask1 = self.decoder_1(torch.cat([up1,down7],dim = 1),torch.cat([up_mask1,down_mask7],dim = 1))

        up2 =  F.interpolate(up1,scale_factor=2)
        up_mask2 =  F.interpolate(up_mask1,scale_factor=2)
        up2,up_mask2 = self.decoder_2(torch.cat([up2,down6],dim = 1),torch.cat([up_mask2,down_mask6],dim = 1))

        up3 =  F.interpolate(up2,scale_factor=2)
        up_mask3 =  F.interpolate(up_mask2,scale_factor=2)
        up3,up_mask3 = self.decoder_3(torch.cat([up3,down5],dim = 1),torch.cat([up_mask3,down_mask5],dim = 1))

        up4 =  F.interpolate(up3,scale_factor=2)
        up_mask4 =  F.interpolate(up_mask3,scale_factor=2)
        up4,up_mask4 = self.decoder_4(torch.cat([up4,down4],dim = 1),torch.cat([up_mask4,down_mask4],dim = 1))

        up5 =  F.interpolate(up4,scale_factor=2)
        up_mask5 =  F.interpolate(up_mask4,scale_factor=2)
        up5,up_mask5 = self.decoder_5(torch.cat([up5,down3],dim = 1),torch.cat([up_mask5,down_mask3],dim = 1))

        up6 =  F.interpolate(up5,scale_factor=2)
        up_mask6 =  F.interpolate(up_mask5,scale_factor=2)
        up6,up_mask6 = self.decoder_6(torch.cat([up6,down2],dim = 1),torch.cat([up_mask6,down_mask2],dim = 1))

        up7 =  F.interpolate(up6,scale_factor=2)
        up_mask7 =  F.interpolate(up_mask6,scale_factor=2)
        up7,up_mask7 = self.decoder_7(torch.cat([up7,down1],dim = 1),torch.cat([up_mask7,down_mask1],dim = 1))

        up8 =  F.interpolate(up7,scale_factor=2)
        up_mask8 =  F.interpolate(up_mask7,scale_factor=2)
        up8,up_mask8 = self.decoder_8(torch.cat([up8,input],dim = 1),torch.cat([up_mask8,mask],dim = 1))

        ## ============== DECODING END =============== ##

        return up8

# =======================  tv_loss ============

'''https://blog.csdn.net/zpkosmos/article/details/105596026'''
'''先看看行不行'''

import torch
import torch.nn as nn
from torch.autograd import Variable
 
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
 
    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])  #算出总共求了多少次差
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()  
        # x[:,:,1:,:]-x[:,:,:h_x-1,:]就是对原图进行错位，分成两张像素位置差1的图片，第一张图片
        # 从像素点1开始（原图从0开始），到最后一个像素点，第二张图片从像素点0开始，到倒数第二个            
        # 像素点，这样就实现了对原图进行错位，分成两张图的操作，做差之后就是原图中每个像素点与相
        # 邻的下一个像素点的差。
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


# ========================= config =====================#
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
ROOT = "."
TRAIN_DIR = f"../input/pix2pix-dataset/maps/maps/train"
VAL_DIR = f"../input/pix2pix-dataset/maps/maps/val"
SAMPLE_DIR = f"{ROOT}/evaluation/"
# print(TRAIN_DIR)


LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 1e-4
BATCH_SIZE = 4
NUM_WORKERS = 0
IMAGE_SIZE = 512
CHANNELS_IMG = 3
L1_LAMBDA_T = 20
L1_LAMBDA_M = 100
TV_WEIGHT = 0.05
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = True
REMAIN_RATE = 0.1#这个是图像还残留多少的比例
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

# both_transform = A.Compose(
#     [A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),], additional_targets={"image0": "image"},
# )

# transform_only_input = A.Compose(
#     [
#         # A.HorizontalFlip(p=0.5),
#         # A.ColorJitter(p=0.2),
#         # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
#         ToTensorV2(),
#     ]
# )

# # 没用上应该
# transform_only_mask = A.Compose(
#     [
#         # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
#         ToTensorV2(),
#     ]
# )
# =================== dataset ==========================#
from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# import config
from torchvision.utils import save_image

class MapDataset(Dataset):
    def __init__(self,root_dir) -> None:
        super().__init__()
        self.rootdir = root_dir
        self.list_files = os.listdir(self.rootdir)
        # print(self.list_files)

    def __len__(self):
        return len(self.list_files)

    # 因为数据集中的图片是两个连接起来的
    # 我们需要把两个拆开来看
    def __getitem__(self, index)  :
        img_file = self.list_files[index]#其实是单独的文件名
        img_path = os.path.join(self.rootdir,img_file)#这里是图片的路径
        image = np.array(Image.open(img_path))
        input_image = image[:,:600,:]#左边的原始图像
        target_image = image[:,600:,:]#右边的目标图像,目前先不用
        #先只用input_image,等到后面找到了数据集再调整
        #我会返回一个完整的图像以及一个mask
        #暂时还不会相乘
        # input_image= torch.from_numpy(input_image).float()
        # input_image = torch.tensor(input_image, dtype=torch.float32)
        transform1 = transforms.Compose([
        transforms.ToTensor(),
        # transforms.ToPILImage(),
        transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
        # transforms.RandomColorJitter(saturation=0.5, p = 0.1),
        
    ])
        # input_image = torch.tensor(input_image, dtype=torch.float32)
        input_image = transform1(input_image)
        # augmentations = both_transform(image = input_image,image0 = target_image)#缩小到512*512
        # input_image, target_image = augmentations['image'],augmentations['image0']

        # input_image = transform_only_input(image = input_image)["image"]
        # target_image = transform_only_mask(image = target_image)["image"]

        

        mask = mask_init(REMAIN_RATE,IMAGE_SIZE,DEVICE)

        return input_image,mask

# ==================== utils ============================#
import torch
from torchvision.utils import save_image

def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x * y ,y)
        # y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x, folder + f"/input_{epoch}.png")
        # save_image(x *y * 0.5 + 0.5, folder + f"/masked_input_{epoch}.png")
#         if epoch == 1:
#             save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

# =================== train ======================= #
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_fn(disc,gen,loader,opt_disc,opt_gen,L1_LOSS_taltal,L1_LOSS_message,bce,TV_Loss):
    loop = tqdm(loader,leave=True)

    for idx,(img,mask)in enumerate(loop):
        img,mask = img.to(DEVICE),mask.to(DEVICE)

        # x是完整的图片，y是mask

        #tain discriminator
        # with torch.cuda.amp.autocast():
        y_fake = gen(img,mask.detach())
        D_real = disc(img,img)
        D_fake = disc (img,y_fake.detach())
        
        D_real_loss = bce(D_real,torch.ones_like(D_real))
        D_fake_loss = bce(D_fake,torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2 

        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()
        # d_scalar.scale(D_loss).backward()
        # d_scalar.step(opt_disc)
        # d_scalar.update()


        #train Generator
        # with torch.cuda.amp.autocast():

        D_fake = disc(img,y_fake)
        G_fake_loss = bce(D_fake,torch.ones_like(D_fake))
        l_taltal = L1_LOSS_taltal(y_fake,img) * L1_LAMBDA_T
        l_message = L1_LOSS_message(y_fake * mask, img * mask) * L1_LAMBDA_M
        tv_loss = TV_Loss(y_fake) * TV_WEIGHT
        G_loss = G_fake_loss + l_taltal + l_message + tv_loss

        opt_gen.zero_grad()
        G_loss.backward(retain_graph=True)
        opt_gen.step()
        # g_scalar.scale(G_loss).backward()
        # g_scalar.step(opt_gen)
        # g_scalar.update()



def main():
    if os.path.exists(SAMPLE_DIR) == False:
        os.mkdir(SAMPLE_DIR)
    disc = Discriminator(in_channels=3).to(DEVICE)
    gen = Generator(in_channels=3).to(DEVICE)
    opt_disc = optim.Adam(disc.parameters(),lr = LEARNING_RATE_D,betas=(0.5,0.999))
    opt_gen = optim.Adam(gen.parameters(),lr = LEARNING_RATE_G,betas=(0.5,0.999))
    BCE = nn.BCEWithLogitsLoss()#听说wgan的loss和patch gan 的结合效果不好
    L1_LOSS_taltal = nn.L1Loss()
    L1_LOSS_message = nn.L1Loss()
    tv_loss = TVLoss()

    if LOAD_MODEL:
        load_checkpoint(CHECKPOINT_GEN,gen,opt_gen,LEARNING_RATE_G)
        load_checkpoint(CHECKPOINT_DISC,disc,opt_disc,LEARNING_RATE_D)

    train_dataset = MapDataset(root_dir = TRAIN_DIR)
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)
    # g_scaler = torch.cuda.amp.GradScaler()
    # disc_scalar = torch.cuda.amp.GradScaler()
    val_dataset = MapDataset(root_dir=VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    for epoch in range (NUM_EPOCHS):
        train_fn(disc,gen,train_loader,opt_disc,opt_gen,L1_LOSS_taltal,L1_LOSS_message,BCE,tv_loss)
        if SAVE_MODEL and epoch % 10 == 0:
            save_checkpoint(gen,opt_gen,filename=CHECKPOINT_GEN)
            save_checkpoint(disc,opt_disc,filename=CHECKPOINT_DISC)
        if epoch %5 == 0:
            save_some_examples(gen,val_loader,epoch,folder=SAMPLE_DIR)



main()







