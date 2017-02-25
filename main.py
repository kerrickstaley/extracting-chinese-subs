#!/usr/bin/env python3
import cv2
import sys
import pyocr
from PIL import Image

LANG='chi_sim'


def main():
  img = cv2.imread('./scene_from_love_me_if_you_dare.png')
  out_img = cv2.inRange(img, (200, 200, 200), (255, 255, 255))
  show_image(out_img)
  pil_img = Image.fromarray(out_img)
  text = get_tool().image_to_string(
    pil_img,
    lang=LANG,
  )
  print(text)


def get_tool():
  tool = pyocr.get_available_tools()[0]
  return tool


def show_image(img):
  cv2.imshow('image', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


if __name__ == '__main__' and not hasattr(sys, 'ps1'):
  main()
