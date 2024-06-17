import numpy as np
import os
import os.path
import cv2


if __name__ == "__main__":
    # colors = np.loadtxt('semseg/data/target/target_colors.txt').astype('uint8')
    # print(colors)

    image_path = 'semseg/exp/target/pspnet50/result/epoch_50/val/ss/color/晴天白天_ROI_1_20240522102528658-3541.png'
    # label_path
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
    # image = np.float32(image)
    print(image)
    print(np.max(image)) # 255

    # print(image.shape) (512, 512, 3)
    # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W