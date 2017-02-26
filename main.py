#!/usr/bin/env python3
import cv2
import sys
import pyocr
from PIL import Image
import numpy as np

LANG='chi_sim'
TEXT_TOP = 810 / 934
TEXT_BOTTOM = 888 / 914


def main():
  cap = cv2.VideoCapture('./love_me_if_you_dare_ep1.ts')
  success = True
  counter = 0
  while success:
    counter += 1
    success, frame = cap.read()
    if counter % 25:
      continue
    processed, text = get_processed_img_and_text(frame)
    print('{}s'.format(counter / 25), text)
    if text:
      show_unprocessed_processed(frame, processed)


def get_processed_img_and_text(img):
  img = crop_to_text_region(img)
  img = threshold(img)
  img = dilate_erode(img)
  # average character is 581 pixels
  if np.count_nonzero(img) < 1000:
    return img, ''
  pil_img = Image.fromarray(img)
  text = get_tool().image_to_string(
    pil_img,
    lang=LANG,
  )
  return img, text


def get_tool():
  tool = pyocr.get_available_tools()[0]
  return tool


def crop_to_text_region(img):
  if len(img.shape) == 3:
    width, height, _ = img.shape
  else:
    width, height = img.shape
  return img[int(width * TEXT_TOP) : int(width * TEXT_BOTTOM), :]


def threshold(img):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  return cv2.inRange(hsv, (0, 0, 170), (179, 25, 255))


def dilate_erode(img):
  "Closes the img"
  kernel = np.ones((5, 5), np.uint8)
  img = cv2.dilate(img, kernel)
  img = cv2.erode(img, kernel)
  return img


def show_image(img):
  cv2.imshow('image', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def show_unprocessed_processed(unp, p):
  cv2.imshow('unprocessed', unp)
  cv2.imshow('processed', p)
  while cv2.waitKey(100) != ord('j'):
    pass

  cv2.destroyAllWindows()


if __name__ == '__main__' and not hasattr(sys, 'ps1'):
  main()
