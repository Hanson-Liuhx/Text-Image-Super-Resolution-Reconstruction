import torch
import math
from torch import nn
import numpy
from torchvision.models.vgg import vgg16
import scipy 
import scipy.misc

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.psnr_loss = PSNRLoss()
        self.ssim_loss = SSIMLoss()
        self.wmse_loss = WMSELoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)

        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))

        # Image Loss 
        image_loss = self.mse_loss(out_images, target_images)

        # TV Loss
        tv_loss = self.tv_loss(out_images)

        # PSNRLoss bigger the better
        psnr_loss = self.psnr_loss(out_images,target_images)

        #SSIMLoss, bigger better, closer to 1
        # ssim_loss = self.ssim_loss(out_images,target_images)
        # print(ssim_loss)
        
        #WMSELoss
        wmse_loss = self.wmse_loss(out_images,target_images)

        # return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss - psnr_loss - ssim_loss
        # return image_loss + 0.001 * adversarial_loss + 2e-8 * tv_loss - psnr_loss - ssim_loss
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss + 0.1 * wmse_loss - 0.2*psnr_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class WMSELoss(nn.Module):
    def __init__(self):
        super(WMSELoss, self).__init__()
    def forward(self,x,y):
        batch_size = x.size()[0]
        channels = x.size()[1]
        G_X = [[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]
        G_Y = [[1, 2, 1],
              [0, 0, 0],
              [-1, -2, -1]]
        X_kernel = torch.FloatTensor(G_X).expand(batch_size,channels,3,3)
        Y_kernel = torch.FloatTensor(G_Y).expand(batch_size,channels,3,3) 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_kernel = X_kernel.to(device)
        Y_kernel = Y_kernel.to(device)
        X_weight = nn.Parameter(data=X_kernel, requires_grad=False)
        Y_weight = nn.Parameter(data=Y_kernel, requires_grad=False)
        train_Gx = torch.nn.functional.conv2d(x,X_weight,padding=1)
        train_Gy = torch.nn.functional.conv2d(x,Y_weight,padding=1)
        ground_Gx = torch.nn.functional.conv2d(y,X_weight,padding=1)
        ground_Gy = torch.nn.functional.conv2d(y,Y_weight,padding=1)
        train_img = (torch.abs(train_Gx)+torch.abs(train_Gy))
        ground_img = (torch.abs(ground_Gx)+torch.abs(ground_Gy))
        # transform  function, three function, f, sqrt(f),pow(f)
        f_train = train_img.sqrt()
        f_ground = ground_img.sqrt()
        # f_train = torch.mul(train_img, train_img)
        # f_ground = torch.mul(ground_img, ground_img)
        # f_train = train_img
        # f_ground = ground_img
        g_train = f_train.sqrt()
        g_ground = f_ground.sqrt()
        mse_loss = nn.MSELoss()

        # print("origin ",x.size())
        # print("before ",train_Gx.size())
        # print("after ",g_train.size())
        # print("first ",g_ground[0][0])
        # print("second ",g_ground[0][1])
        # print("third ", g_ground[0][2])
        # arg_train = g_train.expand(batch_size,channels,g_train.size()[2],g_train.size()[3])
        # arg_ground = g_ground.expand(batch_size,channels,g_ground.size()[2],g_ground.size()[3])
        wse_loss = mse_loss(g_train,g_ground)

        # GX = ground_Gx.cpu().detach().numpy()[0][0]
        # scipy.misc.imsave('/root/SRGAN/srgan_icdar/szz_SRGAN/czydeqingqiu/ground_Gx.png', GX)
        # GY = ground_Gy.cpu().detach().numpy()[0][0]
        # scipy.misc.imsave('/root/SRGAN/srgan_icdar/szz_SRGAN/czydeqingqiu/ground_Gy.png', GY)
        # GT = g_train.cpu().detach().numpy()[0][0]
        # scipy.misc.imsave('/root/SRGAN/srgan_icdar/szz_SRGAN/czydeqingqiu/g_train.png', GT)
        # GG = g_ground.cpu().detach().numpy()[0][0]
        # scipy.misc.imsave('/root/SRGAN/srgan_icdar/szz_SRGAN/czydeqingqiu/g_ground.png', GG)
        # GX.save('/root/SRGAN/srgan_icdar/szz.SRGAN/czydeqingqiu/ground_Gx.png', "png")

        return wse_loss

class PSNRLoss(nn.Module):
    def __init__(self):
        super(PSNRLoss, self).__init__()
    
    def forward(self, x,y):
        #x size [64,3,88,88]
        #y size [64,3,88,88]
        r2y = rgd2yCbCr(x,y)
        new_x = r2y[0]
        new_y = r2y[1]
        mse = (new_y - new_x).norm(2)
        # mse = nn.MSELoss(new_x,new_y)
        # print(mse)
        assert(mse!=0)

        psn_loss = 20 * math.log10(255.0 / mse)
        # print(psn_loss)
        return psn_loss

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
    
    def forward(self, x,y):
        batch_size = x.size()[0]
        width = x.size()[2]
        heigh = x.size()[3]

        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        ssi_loss = 0
        for index in range(batch_size):
            x = gaussion_con(x)
            y = gaussion_con(y)

            one_x_r = x[index,0]
            one_y_r = y[index,0]
            one_x_g = x[index,1]
            one_y_g = y[index,1]
            one_x_b = x[index,2]
            one_y_b = y[index,2]

            mean_x = one_x_r.mean()
            mean_y = one_y_r.mean()
            mean_x += one_x_g.mean()
            mean_y += one_y_g.mean()
            mean_x += one_x_b.mean()
            mean_y += one_y_b.mean()
            mean_x = mean_x/3
            mean_y = mean_y/3
            
            std_x = one_x_r.std()**2
            std_y = one_y_r.std()**2
            std_x += one_x_g.std()**2
            std_y += one_y_g.std()**2
            std_x += one_x_b.std()**2
            std_y += one_y_b.std()**2
            std_x = std_x/3
            std_y = std_y/3
            
            part1 = (one_x_r-mean_x).mul(one_y_r-mean_y)
            part1_sum = 0
            part2 = (one_x_g-mean_x).mul(one_y_g-mean_y)
            part2_sum = 0
            part3 = (one_x_b-mean_x).mul(one_y_b-mean_y)
            part3_sum = 0
            count = -1
            for so in range(width):
                for sd in range(heigh):
                    part1_sum = part1_sum + part1[so,sd]
                    part2_sum = part2_sum + part2[so,sd]
                    part3_sum = part3_sum + part3[so,sd]
                    count = count + 1
                
            assert(count!=0)

            count = count*3
            conv = (part1_sum+part2_sum+part3_sum)/count
            
            part_a = 2*mean_x*mean_y + C1
            part_b = 2*conv + C2
            part_c = mean_x**2 + mean_y**2 + C1
            part_d = std_x**2 + std_y**2 + C2
            one_turn = (part_a*part_b)/(part_c*part_d)
            ssi_loss += one_turn
        return ssi_loss

def rgd2yCbCr(x,y):
    batch_size = x.size()[0]
    width = x.size()[2]
    heigh = x.size()[3]
    # one_img [3,88,88]
    #x : out_image; y : target_image

    # Y = 0.257*R+0.564*G+0.098*B+16
    # Cb = -0.148*R-0.291*G+0.439*B+128
    # Cr = 0.439*R-0.368*G-0.071*B+128
    out1 = torch.ones(batch_size,width,heigh)
    out2 = torch.ones(batch_size,width,heigh)
    for index in range(batch_size):
        one_x_r = x[index,0]
        one_y_r = y[index,0]
        one_x_g = x[index,1]
        one_y_g = y[index,1]
        one_x_b = x[index,2]
        one_y_b = y[index,2]
        out_x = 0.257*one_x_r+0.564*one_x_g+0.098*one_x_b+16
        out_y = 0.257*one_y_r+0.564*one_y_g+0.098*one_y_b+16
        out1[index] = out_x
        out2[index] = out_y
    return [out1,out2]

def gaussion_con(x):
    # kernel=torch.randn(3, 3)
    kernel = [[0.03797616, 0.044863533, 0.03797616],
              [0.044863533, 0.053, 0.044863533],
              [0.03797616, 0.044863533, 0.03797616]]
    x1 = x[0]
    channels=x.size()[1]
    out_channel=channels
    kernel = torch.FloatTensor(kernel).expand(out_channel,channels,3,3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kernel = kernel.to(device)
    weight = nn.Parameter(data=kernel, requires_grad=False)
    # print("weight")
    # print(weight)
    # print("x1 size")
    # print(x1.size())
    # print("weight size")
    # print(weight.size())
    # print("this turn end")
    return torch.nn.functional.conv2d(x,weight,padding=1)              
    
if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
