import os
import numpy as np
import cv2
import math
from scipy import interpolate
from scipy.fft import fft
from enum import Enum
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pylab import *
import skimage as ski

"""
    Steps:
    1. get Image
    2. select ROI(High contrast region)
    3. compute ESF
    4. compute LSF
    5. Get MTF

    Reference: https://github.com/bvnayak/PDS_Compute_MTF

"""


def compute_RGB(image_data: np.ndarray, roi: np.ndarray):
    r_value_list = []
    g_value_list = []
    b_value_list = []
    roi = roi.astype(int)
    # image_data_8bits = (image_data_16bits / 256).astype('uint8')
    image_roi = image_data[roi[0]:roi[1], roi[2]:roi[3]]
    height, width = image_roi.shape[:-1]
    Lab_values = ski.color.rgb2lab(image_roi[:, :, :-1])
    # Lab_values = cv2.cvtColor(image_roi, cv2.COLOR_BGR2LAB)
    Lab_values = np.reshape(Lab_values, [height * width, 3])
    mean_LAB = np.mean(Lab_values, axis=0)

    for i in np.arange(roi[0], roi[1]):
        for j in np.arange(roi[2], roi[3]):
            b_pix, g_pix, r_pix, _ = (image_data[i, j, :])
            r_value_list.append(r_pix)
            g_value_list.append(g_pix)
            b_value_list.append(b_pix)
    mean_color_value = [np.mean(r_value_list), np.mean(g_value_list), np.mean(b_value_list)]
    # plot the rgb value of roi
    fig, axs = plt.subplots(nrows=3, ncols=2, tight_layout=True)

    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    # ax = fig.add_subplot(111)

    # ax.set_xlabel('pixel')

    axs[0, 0].title.set_text("R")
    axs[0, 0].plot(np.arange(height * width), r_value_list, 'r-',
                   np.arange(height * width), mean_color_value[0] * np.ones(height * width))
    axs[0, 0].text(0.9, 0.9, r"mean R value = " +
                   str(mean_color_value[0]), fontsize=9, bbox=bbox, transform=axs[0, 0].transAxes, horizontalalignment='right')

    axs[1, 0].title.set_text("G")
    axs[1, 0].plot(np.arange(height * width), g_value_list, 'g-',
                   np.arange(height * width), mean_color_value[1] * np.ones(height * width))
    axs[1, 0].text(0.9, 0.9, r"mean G value = " +
                   str(mean_color_value[1]), fontsize=9, bbox=bbox, transform=axs[1, 0].transAxes, horizontalalignment='right')

    axs[2, 0].title.set_text("B")
    axs[2, 0].plot(np.arange(height * width), b_value_list, 'b-',
                   np.arange(height * width), mean_color_value[2] * np.ones(height * width))
    axs[2, 0].text(0.9, 0.9, r"mean B value = " +
                   str(mean_color_value[2]), fontsize=9, bbox=bbox, transform=axs[2, 0].transAxes, horizontalalignment='right')
    # plot the lab value of roi

    axs[0, 1].title.set_text("L")
    axs[0, 1].plot(np.arange(height * width), Lab_values[:, 0],
                   np.arange(height * width), mean_LAB[0] * np.ones(height * width))
    axs[0, 1].text(0.9, 0.9, r"mean L value = " +
                   str(mean_LAB[0]), fontsize=9, bbox=bbox, transform=axs[0, 1].transAxes, horizontalalignment='right')

    axs[1, 1].title.set_text("a")
    axs[1, 1].plot(np.arange(height * width), Lab_values[:, 1],
                   np.arange(height * width), mean_LAB[1] * np.ones(height * width))
    axs[1, 1].text(0.9, 0.9, r"mean a value = " +
                   str(mean_LAB[1]), fontsize=9, bbox=bbox, transform=axs[1, 1].transAxes, horizontalalignment='right')

    axs[2, 1].title.set_text("b")
    axs[2, 1].plot(np.arange(height * width), Lab_values[:, 2],
                   np.arange(height * width), mean_LAB[2] * np.ones(height * width))
    axs[2, 1].text(0.9, 0.9, r"mean b value = " +
                   str(mean_LAB[2]), fontsize=9, bbox=bbox, transform=axs[2, 1].transAxes, horizontalalignment='right')

    plt.show()
    return None


class MTF():
    def __init__(self, image_data: np.ndarray, roi: np.ndarray):
        # Input: Grey scale image
        self.image_data = image_data
        self.roi = roi
        self.edge_poly_param = []
        self.esf = []  # edge spread function

    def compute_ESF(self):
        height, width = self.image_data[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]].shape
        image_value_reshaped = np.reshape(self.image_data, height * width)
        distances = np.zeros((width, height))
        column = np.arange(0, height) + 0.5
        for x in range(width):
            distances[x, :] = (self.edge_poly_param[0] * column - (x + 0.5) + self.edge_poly_param[1]
                               ) / np.sqrt(self.edge_poly_param[0] * self.edge_poly_param[0] + 1)
        distances_reshaped = np.reshape(distances, height * width)
        indexes = np.argsort(distances_reshaped)
        if np.average(image_value_reshaped[indexes[:10]]) > np.average(image_value_reshaped[indexes[-10:]]):
            distances_reshaped = distances_reshaped[indexes] * -1
        image_value_reshaped = image_value_reshaped[indexes]
        if distances_reshaped[0] > distances_reshaped[1]:
            distances_reshaped = np.flip(distances_reshaped)
            image_value_reshaped = np.flip(image_value_reshaped)

        self.esf = [distances_reshaped, image_value_reshaped]
        # Plot the esf
        # x = [0, self.image_data[1].shape[1] - 1]
        # y = np.polyval(self.edge_poly_param, x)
        # fig = gcf()
        # fig.canvas.manager.set_window_title('Raw ESF')
        # (ax1, ax2) = plt.subplots(2)
        # ax1.imshow(self.image_data, cmap='gray', vmin=0.0, vmax=1.0)
        # ax1.plot(x, y, color='red')
        # ax2.plot(distances, values)
        # plt.show()
        # plt.show(block=False)

    def filter_ESF(self):
        res = np.unique(self.esf[0], return_index=True, return_counts=True)
        indexes = res[1]
        counts = res[2]
        distances = self.esf[0][indexes]

    def compute_LSF(self):
        return np.diff(self.edgespreadfunction)

    def compute_MTF(self):
        linespreadfunction = self.compute_LSF(self.edgespreadfunction)
        return np.abs(np.fft.fft(linespreadfunction))


class ROI_Select:
    def __init__(self, image_data: np.array):
        self.image_data_original = image_data
        self.image_data = (image_data / 256).astype('uint8')
        # self.image_data_16bits = image_data_16bits
        # self.image_data_8bits = (self.image_data_16bits / 256).astype('uint8')
        self.ax_0 = []
        self.roi = []

    def click_box(self, click_1, click_2):
        # Click_1: The first click defines the starting corner of the bounding box
        # Drag the box
        # Click_2: The second click defines the end corner of the bounding box
        x1, y1 = click_1.xdata, click_1.ydata
        x2, y2 = click_2.xdata, click_2.ydata
        self.roi = np.array([y1, y2, x1, x2])

    def exit_selection(self, event):
        # Press 'enter' and perform the measure of the roi
        if event.key in ['enter']:
            # PDS_Compute_MTF(self.image_data, self.roi)
            _ = compute_RGB(self.image_data_original, self.roi)
            cv2.waitKey(0)

    def select_roi(self):
        _, current_ax = plt.subplots()
        # plt.imshow(self.image_data_8bits, cmap='gray')  # plot grayscale image
        plt.imshow(self.image_data)
        _ = RectangleSelector(current_ax,
                              self.click_box,
                              useblit=True,
                              button=[1, 2, 3],
                              minspanx=5, minspany=5,
                              spancoords='pixels',
                              interactive=True)
        plt.connect('key_press_event', self.exit_selection)
        plt.show()

    def run(self):
        _ = self.select_roi()


if __name__ == '__main__':
    # img_file = 'MasterThesis/data/IGI_Urbanmapper2/RGBI_IMG/2023_02_0136_IGI_RGBI.tif'
    img_file = 'MasterThesis/data/UCE_M3_dataset/RGBI_IMG/2023_02_0067_VEX_RGBI.tif'
    # img_data = cv2.imread(img_file, 0)
    img_data = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    print(img_data.dtype)
    # img_norm = cv2.normalize(img_data, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    img_cropped = img_data[3000:8000, 10000:15000]
    roi = ROI_Select(img_cropped)
    roi.run()
    # img_rescaled = (img_data / 256).astype('uint8')
    # print(img_rescaled.dtype)
    # cv2.imwrite("MasterThesis/Results/IGI-UrbanMapper2/RGBI-IMG-8bits/Rescaled.tif", img_rescaled)
