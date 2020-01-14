from polarcam import *
from utils import *
from lib import *
import pandas as pd
import matplotlib.pyplot as pl
import sys
from scipy.misc import imsave
import rospy
import time
import os
import cv2
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from cv_bridge import CvBridge, CvBridgeError
import datetime
import argparse
import rosbag
import numpy as np


topic_names = ['pola1/raw', 'pola2/raw']

outputs = ['aop', 'dop', 'inten']

inter_method = 'bilinear'

#
# def converter(self, message):
#     try:
#         bridge = CvBridge()
#         cv_image = bridge.imgmsg_to_cv2(message)
#         return cv_image
#
#     except CvBridgeError as e:
#         print(e)
#
#
# def callbackPolarcam(self, data):
#
#     pola_im = converter(data)
#
#     img = Polaim(pola_im, method=inter_method)
#
#     # publishers = []
#     for idx, out in enumerate(outputs_topics_1):
#         image_pub = rospy.Publisher(out, Image)
#         # publishers.append(image_pub)
#
#         if outputs[idx] is 'aop':
#             image_pub.publish(self.bridge.cv2_to_imgmsg(
#                 img.rgb_aop(), encoding="passthrough"))
#
#         if outputs[idx] is 'dop':
#             image_pub.publish(self.bridge.cv2_to_imgmsg(
#                 img.dop, encoding="passthrough"))
#
#         if outputs[idx] is 'inten':
#             image_pub.publish(self.bridge.cv2_to_imgmsg(
#                 img.inte, encoding="passthrough"))
#
#         if outputs[idx] is 's0':
#             image_pub.publish(self.bridge.cv2_to_imgmsg(
#                 img.stokes[0, :, :], encoding="passthrough"))
#
#         if outputs[idx] is 's1':
#             image_pub.publish(self.bridge.cv2_to_imgmsg(
#                 img.stokes[1, :, :], encoding="passthrough"))
#
#         if outputs[idx] is 's2':
#             image_pub.publish(self.bridge.cv2_to_imgmsg(
#                 img.stokes[2, :, :], encoding="passthrough"))


class Convert_Pola():
    def __init__(self, topic, channel):
        self.channel = channel
        self.topic = topic

        rospy.init_node(
            f'image_convert_{self.topic}_{self.channel}', anonymous=True)

        rospy.on_shutdown(self.shutdown)

        self.pub = rospy.Publisher(f'{self.topic}/{self.channel}', Image)

        rospy.Subscriber(self.topic, Image, self.callback)

    def callback(self, data):
        pola_im = self.converter(data)

        img = Polaim(pola_im, method=inter_method)

        if self.channel is 'aop':
            self.pub.publish(self.bridge.cv2_to_imgmsg(
                img.rgb_aop(), encoding="passthrough"))

        elif self.channel is 'dop':
            self.pub.publish(self.bridge.cv2_to_imgmsg(
                img.dop, encoding="passthrough"))

        elif self.channel is 'inten':
            self.pub.publish(self.bridge.cv2_to_imgmsg(
                img.inte, encoding="passthrough"))

        elif self.channel is 's0':
            self.pub.publish(self.bridge.cv2_to_imgmsg(
                img.stokes[0, :, :], encoding="passthrough"))

        elif self.channel is 's1':
            self.pub.publish(self.bridge.cv2_to_imgmsg(
                img.stokes[1, :, :], encoding="passthrough"))

        elif self.channel is 's2':
            self.pub.publish(self.bridge.cv2_to_imgmsg(
                img.stokes[2, :, :], encoding="passthrough"))

        elif self.channel is 'i0':
            self.pub.publish(self.bridge.cv2_to_imgmsg(
                img.images[0, :, :], encoding="passthrough"))

        elif self.channel is 'i45':
            self.pub.publish(self.bridge.cv2_to_imgmsg(
                img.images[1, :, :], encoding="passthrough"))

        elif self.channel is 'i90':
            self.pub.publish(self.bridge.cv2_to_imgmsg(
                img.images[2, :, :], encoding="passthrough"))

        elif self.channel is 'i135':
            self.pub.publish(self.bridge.cv2_to_imgmsg(
                img.images[3, :, :], encoding="passthrough"))

        elif self.channel is 'hsl':
            self.pub.publish(self.bridge.cv2_to_imgmsg(
                img.rgb_pola(), encoding="passthrough"))

    def converter(self, message):
        try:
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(message)
            return cv_image

        except CvBridgeError as e:
            print(e)

    def shutdown(self):
        rospy.loginfo('Stopping the Node...')
        rospy.sleep(1)


if __name__ == '__main__':
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            # nodes = []
            for topic in topic_names:
                for channel in outputs:
                    new_node = Convert_Pola(topic, channel)
                    # nodes.append(new_nodes)

            rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Converter Nodes Terminated ...')
