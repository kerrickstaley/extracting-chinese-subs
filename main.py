#!/usr/bin/env python3
from argparse import ArgumentParser
import cv2
import os
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
parser.add_argument('video_file', nargs='?')


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
    model = E0()
    text = model.extract(frame)
    if args.dump_text:
      if text:
        print(text)
    else:
      print('{}s'.format(frame_idx / 25), text)
      if text:
        if args.dump_test_cases:
          cv2.imwrite('test_frame_{}__{}.png'.format(frame_idx, text), frame)
        else:
          show_unprocessed_processed(frame, model.cleaned)


class TextExtractor:
  def __init__(self, debug=False):
    self.debug = debug

  def extract(self, img):
    """
    :param numpy.array img: frame of video
    :return str: extracted subtitle text ('' if there is no subtitle)
    """
    self.cleaned = self.clean_image(img)
    self.raw_text = self.run_ocr(self.cleaned)
    return self.post_process_text(self.raw_text)

  def clean_image(self, img):
    """
    :param numpy.array img: frame of video
    :return numpy.array cleaned: cleaned image, ready to run through OCR
    """
    raise NotImplementedError

  def post_process_text(self, text):
    """
    :param str text: text returned by OCR step
    :return str: cleaned text
    """
    if not text:
      return ''

    # hack: tesseract interprets 一 as _
    new_text = [text[0]]
    for before, mid, after in ngroupwise(3, text):
      if mid == '_' and unicodedata.category(before) == unicodedata.category(after) == 'Lo':
        new_text.append('一')
      else:
        new_text.append(mid)
    new_text.append(text[-1])
    txt = ''.join(new_text)

    # strip out non-Chinese characters
    rv = []
    for c in txt:
      if unicodedata.category(c) != 'Lo':
        continue
      rv.append(c)

    return ''.join(rv)

  def run_ocr(self, img):
    """
    :param numpy.array img: cleaned image
    :return str: extracted subtitle text ('' if there is no subtitle)
    """
    # average character is 581 pixels
    if np.count_nonzero(img) < 1000:
      return ''

    tool = pyocr.get_available_tools()[0]
    pil_img = Image.fromarray(img)
    return tool.image_to_string(
        pil_img,
        lang=LANG,
      )


class E0(TextExtractor):
  def clean_image(self, img):
    if len(img.shape) == 3:
      width, height, _ = img.shape
    else:
      width, height = img.shape
    cropped = img[int(width * TEXT_TOP): int(width * TEXT_BOTTOM), :]
    return self.clean_after_crop(cropped)

  def clean_after_crop(self, cropped):
    img = threshold(cropped)
    img = dilate_erode3(img)
    img = dilate3(img)
    img = img & dilate_erode5(cv2.Canny(cropped, 400, 600))
    return img


def ngroupwise(n, iterable):
  # generalization of the "pairwise" recipe
  iterators = list(itertools.tee(iterable, n))
  for i in range(n):
    for j in range(i):
      next(iterators[i], None)

  return zip(*iterators)


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
  if cv2.waitKey(0) == ord('q'):
    raise Exception('quitting')
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


def get_all_test_frames():
  for dirpath, dirnames, filenames in os.walk('test_frames'):
    if 'unprocessed' in dirpath.split('/'):
      continue
    for filename in filenames:
      yield os.path.join(dirpath, filename)


def test_all():
  passes = 0
  cases = 0
  for fname in get_all_test_frames():
    passes += test_case(fname)
    cases += 1

  print('==== passed {} / {} tests ({} %) ===='.format(
    passes, cases, int(round(passes / cases * 100))))


def test_case(fname, debug=False):
  img = cv2.imread(fname)
  expected_text = fname.split('__')[1][:-4]
  model = E0()
  actual_text = model.extract(img)
  # from IPython import embed; embed()
  inital = pad_string('file {}:'.format('/'.join(fname.split('/')[-2:])), 60)
  print(inital, end='')
  if actual_text == expected_text:
    print('PASSED')
    if not debug:
      return True
  else:
    print("FAILED; got '{}' expected '{}'".format(actual_text, expected_text))
    if not debug:
      return False

  show_unprocessed_processed(img, model.cleaned)


if __name__ == '__main__' and not hasattr(sys, 'ps1'):
  main(parser.parse_args())
