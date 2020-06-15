import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model import Generator

import cv2
import numpy
import ocr

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
# parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
parser.add_argument('--model_name', default='netG_epoch_4_1.pth', type=str, help='generator model epoch name')
parser.add_argument('--ocr', type=int, default=0, choices=[0, 1], help='1, 0')

opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name
ocr_do = opt.ocr
if ocr_do == 1:
    do_ocr = True
else:
    do_ocr = False

results = {'psnr': [], 'ssim': []}

model = Generator(UPSCALE_FACTOR).eval()
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

# test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
# test_set = TestDatasetFromFolder('../../../ICDAR2015-TextSR-dataset/RELEASE_2015-08-31/DATA/TEST', upscale_factor=UPSCALE_FACTOR)
test_set = TestDatasetFromFolder('../../../../ICDAR2015-TextSR-dataset/RELEASE_2015-08-31/DATA/TEST', upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

out_path = 'benchmark_results/SRF_' + str(UPSCALE_FACTOR) + '/'
if not os.path.exists(out_path):
    os.makedirs(out_path)
# with torch.no_grad():

if do_ocr:
    ocrlog = open("ocrlog" + MODEL_NAME + ".txt", "a")
index = 1
total_sr_acc = 0
# total_hr_acc = 0
total_hr_restore_acc = 0

for image_name, lr_image, hr_restore_img, hr_image in test_bar:
    image_name = image_name[0]
    lr_image = Variable(lr_image, volatile=True)
    hr_image = Variable(hr_image, volatile=True)
    if torch.cuda.is_available():
        lr_image = lr_image.cuda()
        hr_image = hr_image.cuda()
    # print(lr_image.size())
    sr_image = model(lr_image)
    # print(sr_image.size())
    # print(hr_image.size())
    mse = ((hr_image - sr_image) ** 2).data.mean()
    psnr = 10 * log10(1 / mse)
    ssim = pytorch_ssim.ssim(sr_image, hr_image).data[0]

    test_images = torch.stack(
        [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
        display_transform()(sr_image.data.cpu().squeeze(0))])
    image = utils.make_grid(test_images, nrow=3, padding=5)
    utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
                    image_name.split('.')[-1], padding=5)

    
    # save psnr\ssim
    # print(image_name)
    results['psnr'].append(psnr)
    results['ssim'].append(ssim)

    #-------for ocr:
    if(do_ocr):
        image1 = utils.make_grid(display_transform()(hr_restore_img.squeeze(0)), nrow=3, padding=5)
        utils.save_image(image1, out_path + 'index_%d_hr_restore.png' % (index), padding=5)

        image2 = utils.make_grid(display_transform()(hr_image.data.cpu().squeeze(0)), nrow=3, padding=5)
        utils.save_image(image2, out_path + 'index_%d_hr.png' % (index), padding=5)

        image3 = utils.make_grid(display_transform()(sr_image.data.cpu().squeeze(0)), nrow=3, padding=5)
        utils.save_image(image3, out_path + 'index_%d_sr.png' % (index), padding=5)
        
        hr_restore = cv2.imread(out_path + 'index_%d_hr_restore.png' % (index), 0)
        hr = cv2.imread(out_path + 'index_%d_hr.png' % (index), 0)
        sr = cv2.imread(out_path + 'index_%d_sr.png' % (index), 0)
        sr_acc, hr_restore_acc = ocr.getAccuracy(sr, hr, hr_restore, index)
        total_sr_acc += sr_acc
        # total_hr_acc += hr_acc
        total_hr_restore_acc += hr_restore_acc
    index += 1

out_path = 'statistics/'
saved_results = {'psnr': [], 'ssim': []}
for item in results.values():
    psnr = np.array(results['psnr'])
    ssim = np.array(results['ssim'])
    if (len(psnr) == 0) or (len(ssim) == 0):
        psnr = 'No data'
        ssim = 'No data'
    else:
        psnr = psnr.mean()
        ssim = ssim.mean()
    saved_results['psnr'].append(psnr)
    saved_results['ssim'].append(ssim)

data_frame = pd.DataFrame(saved_results, results.keys())
data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_test_results.csv', index_label='DataSet')

if(do_ocr):
    average_sr_acc = total_sr_acc / (index - 1)
    # average_hr_acc = total_hr_acc / (index - 1)
    average_hr_restore_acc = total_hr_restore_acc / (index - 1)
    print("Evaluating with OCR")
    ocrres = "Average GEN accuracy: "+str(average_sr_acc)[:8]+"\nAverage LRI accuracy: "+str(average_hr_restore_acc)[:8]+'\n\n'
    ocrlog.write(ocrres)
    ocrlog.close()