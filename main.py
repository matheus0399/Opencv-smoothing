import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

input_imgs = ['test.png']
for count in range(len(input_imgs)):
    img = cv.imread(input_imgs[count], 0)
    img = img/255
    img = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA)

    cv.imshow('{} original image'.format(input_imgs[count]), img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    img_hist = img * 255
    plt.hist(img_hist.ravel(), 256, [0, 256])
    plt.show()

    x, y = img.shape
    noise_img = np.zeros((x, y), dtype=np.float32)

    # salt and pepper amount

    pepper = 0.05
    salt = 1 - pepper

    # create salt and peper noise image
    for i in range(x):
        for j in range(y):
            rdn = np.random.random()
            if rdn < pepper:
                noise_img[i][j] = 0
            elif rdn > salt:
                noise_img[i][j] = 1
            else:
                noise_img[i][j] = img[i][j]

    cv.imshow('{} with noise'.format(input_imgs[count]), noise_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    img_hist = noise_img * 255
    plt.hist(img_hist.ravel(), 256, [0, 256])
    plt.show()
    cv.imwrite('{}_noise.png'.format(input_imgs[count]), noise_img * 255)

    img_median_filter = cv.medianBlur(noise_img, 3)

    cv.imshow('{} with median filter'.format(input_imgs[count]), img_median_filter)
    cv.imwrite('{}_median.png'.format(input_imgs[count]), img_median_filter * 255)

    cv.waitKey(0)
    cv.destroyAllWindows()
    img_hist = img_median_filter * 255
    plt.hist(img_hist.ravel(), 256, [0, 256])
    plt.show()

    g = cv.GaussianBlur(noise_img, (17, 17), 0)

    cv.imshow('{} with low-pass filter'.format(input_imgs[count]), g)
    cv.imwrite('{}_passa-baixa.jpg'.format(input_imgs[count]), g * 255)

    cv.waitKey(0)
    cv.destroyAllWindows()
    img_hist = g * 255
    plt.hist(img_hist.ravel(), 256, [0, 256])
    plt.show()

    ## TRANSFOMADA FOURIER
    ##titles = ['{} original Fourier'.format(input_imgs[count]), '{} noise Fourier'.format(input_imgs[count]), '{} median Fourier'.format(input_imgs[count]), '{} low-pass Fourier'.format(input_imgs[count])]
    ##imgs = [img, noise_img, img_median_filter, g]

    ##for i in range(4):
    ##    f = np.fft.fft2(imgs[i])
    ##    fshift = np.fft.fftshift(f)
    ##    magnitude_spectrum = 20*np.log(np.abs(fshift))
    ##    magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
    ##    img_and_magnitude1 = np.concatenate((imgs[i], magnitude_spectrum), axis=1)

    ##    cv.imshow(titles[i], img_and_magnitude1)

    ##cv.waitKey(0)
    ##cv.destroyAllWindows()