import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def comparison_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def comparison_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


if __name__ == '__main__':
    ori = cv2.imread('Data/NoNoise/1.png')
    cp_psnr = []
    cp_ssim = []
    # cp_psnr.append(comparison_psnr(ori, ori))
    for j in range(3, 11, 2):
        com = cv2.imread('Data/NoiseLevel{}/1.png'.format(j))
        # Note: PSNR ^ -> similarity ^
        cp_psnr.append(comparison_psnr(ori, com))
        cp_ssim.append(comparison_ssim(ori, com))
        print("{}: PSNR value is {} dB".format(j, cp_psnr[-1]))
        print("\tSSIM value is {}".format(cp_ssim[-1]))
    fig_1 = plt.figure()
    plt.title('Comparison PSNR of 1.png')
    plt.xlabel('File Index')
    plt.ylabel('PSNR (dB)')
    ax = fig_1.add_subplot(111)
    ax.bar(range(len(cp_psnr)), cp_psnr, color='green')
    plt.show()
    fig_2 = plt.figure()
    plt.title('Comparison SSIM of 1.png')
    plt.xlabel('File Index')
    plt.ylabel('SSIM level')
    ax = fig_2.add_subplot(111)
    ax.bar(range(len(cp_ssim)), cp_ssim, color='blue')
    plt.show()
