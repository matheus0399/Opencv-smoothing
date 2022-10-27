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

    [M, N] = noise_img.shape
    FT_img = np.fft.fft2(noise_img)
    n = 4
    D0 = 70
    u = np.arange(M)
    v = np.arange(N)

    idx = np.argwhere(u > M / 2)
    for i in idx:
        u[i] = u[i] - M
    idy = np.argwhere(v > N / 2)
    for i in idy:
        v[i] = v[i] - N

    [V, U] = np.meshgrid(v, u)
    D = np.sqrt(np.power(U, 2) + np.power(V, 2))
    H = 1. / (1 + np.power((D / D0), (2 * n)))
    G = H * FT_img
    butter = np.real(np.fft.ifft2(G))

    cv.imshow('{} with butter'.format(input_imgs[count]), butter)
    cv.imwrite('{}_butterworth.png'.format(input_imgs[count]), butter * 255)

    cv.waitKey(0)
    cv.destroyAllWindows()
    img_hist = butter * 255
    plt.hist(img_hist.ravel(), 256, [0, 256])
    plt.show()
