#!/usr/bin/env python3
from argparse import ArgumentParser
import cv2
import sys
import pyocr
from PIL import Image
import numpy as np
import glob
import unicodedata
import itertools

LANG='chi_sim'
TEXT_TOP = 810 / 934
TEXT_BOTTOM = 888 / 914

parser = ArgumentParser(description='extract subtitles')
parser.add_argument('--dump-test-cases', action='store_true')
parser.add_argument('--test-all', action='store_true')
parser.add_argument('--test')
parser.add_argument('--dump-text', action='store_true')
parser.add_argument('video_file')


def main(args):
  if args.test_all:
    test_all()
    return
  if args.test:
    test_case(args.test, debug=True)
    return
  cap = cv2.VideoCapture(args.video_file)
  success = True
  frame_idx = -1
  while success:
    frame_idx += 1
    success, frame = cap.read()
    if frame_idx % 25:
      continue
    processed, text = get_processed_img_and_text(frame)
    if args.dump_text:
      if text:
        print(text)
    else:
      print('{}s'.format(frame_idx / 25), text)
      if text:
        if args.dump_test_cases:
          cv2.imwrite('test_frame_{}__{}.png'.format(frame_idx, text), frame)
        else:
          show_unprocessed_processed(frame, processed)


def get_processed_img_and_text(img):
  cropped = crop_to_text_region(img)
  img = threshold(cropped)
  img = dilate_erode3(img)
  img = dilate3(img)
  img = img & dilate_erode5(cv2.Canny(cropped, 400, 600))
  # average character is 581 pixels
  if np.count_nonzero(img) < 1000:
    return img, ''
  pil_img = Image.fromarray(img)
  text = get_tool().image_to_string(
    pil_img,
    lang=LANG,
  )
  text = strip_non_chinese_characters(text)
  return img, text


def ngroupwise(n, iterable):
  # generalization of the "pairwise" recipe
  iterators = list(itertools.tee(iterable, n))
  for i in range(n):
    for j in range(i):
      next(iterators[i], None)

  return zip(*iterators)


def strip_non_chinese_characters(txt):
  if not txt:
    return ''
  # hack: tesseract interprets 一 as _
  new_txt = [txt[0]]
  for before, mid, after in ngroupwise(3, txt):
    if mid == '_' and unicodedata.category(before) == unicodedata.category(after) == 'Lo':
      new_txt.append('一')
    else:
      new_txt.append(mid)
  new_txt.append(txt[-1])
  txt = ''.join(new_txt)

  rv = []
  for c in txt:
    if unicodedata.category(c) != 'Lo':
      continue
    rv.append(c)

  return ''.join(rv)


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


def dilate_erode5(img):
  "Closes the img"
  kernel = np.ones((5, 5), np.uint8)
  img = cv2.dilate(img, kernel)
  img = cv2.erode(img, kernel)
  return img


def dilate_erode3(img):
  "Closes the img"
  kernel = np.ones((3, 3), np.uint8)
  img = cv2.dilate(img, kernel)
  img = cv2.erode(img, kernel)
  return img


def dilate3(img):
  kernel = np.ones((3, 3), np.uint8)
  return cv2.dilate(img, kernel)


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


def pad_string(s, l):
  chars_taken = len(s)
  for c in s:
    if unicodedata.east_asian_width(c) == 'W':
      chars_taken += 1

  return s + ' ' * (l - chars_taken)


def test_all():
  passes = 0
  cases = 0
  for fname in sorted(glob.glob('test_frames/*.png')):
    passes += test_case(fname)
    cases += 1

  print('==== passed {} / {} tests ({} %) ===='.format(
    passes, cases, int(round(passes / cases * 100))))


def test_case(fname, debug=False):
  img = cv2.imread(fname)
  expected_text = fname.split('__')[1][:-4]
  processed, actual_text = get_processed_img_and_text(img)
  # from IPython import embed; embed()
  inital = pad_string('file {}:'.format(fname.split('/')[-1]), 60)
  print(inital, end='')
  if actual_text == expected_text:
    print('PASSED')
    if not debug:
      return True
  else:
    print("FAILED; got '{}' expected '{}'".format(actual_text, expected_text))
    if not debug:
      return False

  show_unprocessed_processed(img, processed)


if __name__ == '__main__' and not hasattr(sys, 'ps1'):
  main(parser.parse_args())
