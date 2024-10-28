import os 
import cv2
import numpy as np
from skimage.metrics import structural_similarity as cal_ssim
import glob

def get_file_name(path):
    basename = os.path.basename(path)
    onlyname = os.path.splitext(basename)[0]
    return onlyname


total_psnr = 0
total_ssmi = 0


# dataset = 'ihaze'
# dataset = 'ohaze'
# dataset = 'densehaze'
dataset = 'nhhaze'

if dataset in ['ihaze', 'ohaze']: 
    ext = 'jpg'

if dataset in ['densehaze', 'nhhaze']: 
    ext = 'png'


testset = glob.glob(f'datasets/{dataset}/val_B/*.{ext}')

output_folder = f"outputs/NH_results"
txtfile = open(f"./{output_folder}/test_results.txt", "w+")

for path in testset:

	fname = get_file_name(path)

	pred = cv2.imread(f"./{output_folder}/{fname}_hazy.png")
	gt = cv2.imread(path)

	psnr = cv2.PSNR(pred, gt)
	ssmi = cal_ssim(gt, pred, data_range=pred.max() - pred.min(), multichannel=True)

	print(fname, psnr, ssmi)
	print(fname, psnr, ssmi, file=txtfile)

	total_psnr += psnr
	total_ssmi += ssmi

average_psnr = total_psnr/len(testset)
average_ssmi = total_ssmi/len(testset)

print("Avg. PSNR:", average_psnr)
print("Avg. SSMI:", average_ssmi)

print("Avg. PSNR:", average_psnr, file=txtfile)
print("Avg. SSMI:", average_ssmi, file=txtfile)