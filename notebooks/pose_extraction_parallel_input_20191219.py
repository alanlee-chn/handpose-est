## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path
# Import for hand pose estimation
import sys
sys.path.insert(0, '../python')
sys.path.insert(0, '../')
import torch
import util
from model import handpose_model
import util
import math
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
from torchvision import transforms
from skimage.measure import label

input_address = "../../../Documents/20191219_141054.bag"
#input_address = "/mnt/RAM_disk/20191219_141054.bag"
use_recording = True
repeat_recording = False
im_show = True
write2file = False
batch_size = 1
if write2file:
    text_file = open("../../../Documents/Output.txt", "w")
    text_file.write("No\tPoint0x\tPoint0y\tPoint0z\tPoint1x\tPoint1y\tPoint1z\tPoint5x\tPoint5y\tPoint5z\tPoint9x\tPoint9y\tPoint9z\tPoint13x\tPoint13y\tPoint13z\tPoint17x\tPoint17y\tPoint17z\n")
# Create a pipeline
pipeline = rs.pipeline()
#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
# Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
if use_recording:
    rs.config.enable_device_from_file(config, input_address, repeat_playback=repeat_recording)

config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 15)

# Start streaming
profile = pipeline.start(config)

# Allow the recording to be played frame by frame
if use_recording:
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# preparation work for hand pose estimation
model = handpose_model()
model_dict = torch.load('../model/hand_pose_model.pth')
model.load_state_dict(util.transfer(model, model_dict))
# since the input data is loaded in cuda, the model should be loaded in cuda
model = model.cuda()
no_pic=0
# Streaming loop
try:
    while True:
        if write2file:
            text_file.write(str(no_pic)+'\t')

        batch_no = 0
        while batch_no < batch_size:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue
            else:
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                if batch_no == 0:
                    depth_image_array = depth_image[:, :, np.newaxis]
                    color_image_array = color_image[:, :, :, np.newaxis]
                else:
                    depth_image_array = np.concatenate((depth_image_array, depth_image[:, :, np.newaxis]), axis=2)
                    color_image_array = np.concatenate((color_image_array, color_image[:, :, :, np.newaxis]), axis=3)
                batch_no = batch_no + 1
        #depth_image = np.asanyarray(aligned_depth_frame.get_data())
        #color_image = np.asanyarray(color_frame.get_data())

        # This is where the pose estimation starts
        oriImg = color_image_array
        scale_search = [0.5, 1.0, 1.5, 2.0]
        boxsize = 368
        stride = 8
        padValue = 128
        thre = 0.05
        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 22, oriImg.shape[3]))
        for m in range(len(multiplier)):
            scale = multiplier[m]
            for q in range(oriImg.shape[3]):
                imageToTest = cv2.resize(oriImg[:, :, :, q], (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)
                im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
                im = np.ascontiguousarray(im)
                if q == 0:
                    im_array = np.asarray(im)
                else:
                    im_array = np.concatenate((im_array, im), axis=0)
            #imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            #imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)
            #im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
            #im = np.ascontiguousarray(im)
            data = torch.from_numpy(im_array).float()
            if torch.cuda.is_available():
                data = data.cuda()
            # data = data.permute([2, 0, 1]).unsqueeze(0).float()
            with torch.no_grad():
                output = model(data).cpu().numpy()
            for q in range(output.shape[0]):
                heatmap = np.transpose(np.squeeze(output[q, :, :, :]), (1, 2, 0))
                heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
                heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
                heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
                heatmap_avg[:, :, :, q] = heatmap_avg[:, :, :, q] + heatmap[:, :, :] / len(multiplier)
            # extract outputs, resize, and remove padding
            #heatmap = np.transpose(np.squeeze(output), (1, 2, 0))  # output 1 is heatmaps
            #heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            #heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            #heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
            #heatmap_avg += heatmap / len(multiplier)

        for q in range(output.shape[0]):
            all_peaks = []
            for part in range(21):
                map_ori = heatmap_avg[:, :, part, q]
                one_heatmap = gaussian_filter(map_ori, sigma=3)
                # all the point with value larger than the threshold
                binary = np.ascontiguousarray(one_heatmap > thre, dtype=np.uint8)
                # 全部小于阈值
                if np.sum(binary) == 0:
                    all_peaks.append(-1)
                    continue
                label_img, label_numbers = label(binary, return_num=True, connectivity=binary.ndim)
                # find the index representing the biggest clutter of label_img == i
                max_index = np.argmax([np.sum(map_ori[label_img == i]) for i in range(1, label_numbers + 1)]) + 1
                # set none max image to 0
                label_img[label_img != max_index] = 0
                map_ori[label_img == 0] = 0

                # get the coordinates of the peak of the largest clutter of one key point
                def npmax(array):
                    arrayindex = array.argmax(1)
                    arrayvalue = array.max(1)
                    i = arrayvalue.argmax()
                    j = arrayindex[i]
                    return i, j

                y, x = npmax(map_ori)
                all_peaks.append((x, y))

            edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
                     [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

            plt.imshow(oriImg[:, :, [0, 1, 2], q])

            for i, cell in enumerate(all_peaks):
                if cell == -1:
                    if write2file:
                        if i == 0 or i == 1 or i == 5 or i == 9 or i == 13 or i == 17:
                            text_file.write(str(-1) + '\t' + str(-1) + '\t' + str(-1))
                            if i == 17:
                                text_file.write('\n')
                            else:
                                text_file.write('\t')
                        # print(i, ", ")
                else:
                    if i == 0 or i == 1 or i == 5 or i == 9 or i == 13 or i == 17:
                        (x, y) = cell
                        plt.plot(x, y, 'r.')
                        dist = depth_image_array[y, x, q]
                        #dist = depth_image[y, x]
                        if write2file:
                            text_file.write(str(x) + '\t' + str(y) + '\t' + str(dist))
                        # plt.text(x, y, str(i)+'<'+str(dist)+'>')
                        plt.text(x, y, str(i))
                        if write2file and i == 17:
                            text_file.write('\n')
                        elif write2file:
                            text_file.write('\t')
            plt.axis('off')
            if im_show:
                plt.show()
            else:
                plt.savefig('/home/lee/Documents/pics/' + str(no_pic) + '.png')
                plt.clf()
            no_pic = no_pic + 1

finally:
    pipeline.stop()
    if write2file:
        text_file.close()
    #del model