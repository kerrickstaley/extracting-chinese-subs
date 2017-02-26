#!/usr/bin/env python3
import cv2
import sys
import pyocr
from PIL import Image

LANG='chi_sim'
TEXT_TOP = 810 / 934
TEXT_BOTTOM = 888 / 914


def main():
  img = cv2.imread('./scene_from_love_me_if_you_dare.png')
  img = crop_to_text_region(img)
  img = threshold(img)
  show_image(img)
  pil_img = Image.fromarray(img)
  text = get_tool().image_to_string(
    pil_img,
    lang=LANG,
  )
  print(text)


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
  return cv2.inRange(img, (180, 180, 180), (255, 255, 255))


def show_image(img):
  cv2.imshow('image', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


if __name__ == '__main__' and not hasattr(sys, 'ps1'):
  main()
