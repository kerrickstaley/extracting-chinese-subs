#!/usr/bin/env python3
import inspect
import itertools
import os
import sys
import unicodedata
from argparse import ArgumentParser

import cv2
import numpy as np
import pyocr
from PIL import Image

LANG='chi_sim'

parser = ArgumentParser(description='extract subtitles')
parser.add_argument('--dump-test-cases', action='store_true')
parser.add_argument('--test-all', action='store_true')
parser.add_argument('--test')
parser.add_argument('--dump-text', action='store_true')
parser.add_argument('--cmp-old', help='old model to compare')
parser.add_argument('--cmp-new', help='new model to compare')
parser.add_argument('--model', help='model to use', default='e2')
parser.add_argument('--debug', help='debug model', action='store_true')
parser.add_argument('video_file', nargs='?')


def main(args):
  model_class = MODELS[args.model]
  if args.test_all:
    test_all(model_class)
    return
  if args.test:
    test_case(model_class, args.test, debug=args.debug)
    return
  if args.cmp_old:
    compare_models(MODELS[args.cmp_old], MODELS[args.cmp_new])
    return
  cap = cv2.VideoCapture(args.video_file)
  success = True
  frame_idx = -1
  while success:
    frame_idx += 1
    success, frame = cap.read()
    if frame_idx % 25:
      continue
    model = model_class()
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


def compare_models(old_mod_class, new_mod_class):
  for fname in get_all_test_frames():
    img = cv2.imread(fname)
    expected_text = fname.split('__')[1][:-4]
    old_mod = old_mod_class()
    old_text = old_mod.extract(img)
    new_mod = new_mod_class()
    new_text = new_mod.extract(img)
    old_pass = old_text == expected_text
    new_pass = new_text == expected_text

    inital = pad_string('file {}:'.format('/'.join(fname.split('/')[-2:])), 60)
    print(inital, end='')
    if old_pass and new_pass:
      print('both pass')
    elif not old_pass and not new_pass:
      if old_text == new_text:
        print('both fail')
      else:
        print('both fail, change from {} to {}'.format(old_text, new_text))
    elif old_pass:
      print('NEW FAILS, new: {}'.format(new_text))
      # show_unprocessed_processed(img, new_mod.cleaned)
      # show_unprocessed_processed(img, old_mod.cleaned)
    else:
      print('OLD FAILS, continuing')


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
  TEXT_TOP = 810 / 934
  TEXT_BOTTOM = 888 / 914
  TEXT_LEFT = 250 / 1280  # min observed was 300 pixels in, each char is 50 pixels wide
  TEXT_RIGHT = 1030 / 1280  # max observed was 300 pixels in from the right

  def clean_image(self, img):
    if len(img.shape) == 3:
      height, width, _ = img.shape
    else:
      height, width = img.shape
    # TODO this is weird
    cropped = img[
        int(height * self.TEXT_TOP): int(height * self.TEXT_BOTTOM),
        int(width * self.TEXT_LEFT): int(width * self.TEXT_RIGHT)]
    return self.clean_after_crop(cropped)

  def clean_after_crop(self, cropped):
    img = threshold(cropped)
    img = dilate_erode3(img)
    img = dilate3(img)
    img = img & dilate_erode5(cv2.Canny(cropped, 400, 600))
    return img


class E1(E0):
  def clean_after_crop(self, cropped):
    self.sharpened = img = sharpen(cropped)
    if self.debug:
      show_image(self.sharpened)
    self.thresholded = img = threshold(img, min_value=191)
    if self.debug:
      show_image(self.thresholded)
    self.canny_mask = cv2.Canny(cropped, 400, 600)
    self.canny_mask = dilate(self.canny_mask, 5)
    self.canny_mask = erode(self.canny_mask, 5)
    img = img & self.canny_mask
    if self.debug:
      show_image(self.canny_mask)
      show_image(img)
    img = remove_small_islands(img)
    img = dilate3(img)
    return img


class E2(E1):
  def get_border_floodfill_mask(self):
    mask = np.zeros(self.thresholded.shape)
    _, contours, hierarchy = cv2.findContours(self.thresholded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for root_idx, contour in enumerate(contours):
      left, top, width, height = cv2.boundingRect(contour)
      right = left + width
      bottom = top + height
      if not (top <= 4 or left <= 4
              or bottom >= self.thresholded.shape[0] - 5 or right >= self.thresholded.shape[1] - 5):
        continue

      cv2.fillPoly(mask, pts=[contour], color=(255, 255, 255))
      for child_contour, (_, _, _, parent_idx) in zip(contours, hierarchy[0]):  # TODO no idea why we have to do [0]
        if parent_idx != root_idx:
          continue
        cv2.fillPoly(mask, pts=[child_contour], color=(0, 0, 0))

    # because we do a dilate3 in super().clean_after_crop, we also need to do that here so the mask matches when we
    # subtract
    mask = dilate(mask, 3)

    return mask

  def clean_after_crop(self, cropped):
    img = super().clean_after_crop(cropped)
    self.border_floodfill_mask = self.get_border_floodfill_mask()
    if self.debug:
      show_image(self.border_floodfill_mask)
    return img - self.get_border_floodfill_mask()


class E3(E2):
  def get_border_floodfill_mask(self):
    h, w = self.thresholded.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    border_points = []
    for r in range(5):
      for c in range(w):
        # top border
        border_points.append((r, c))
        # bottom border
        border_points.append((h - 1 - r, c))
    for c in range(5):
      for r in range(h):
        # left border
        border_points.append((r, c))
        # right border
        border_points.append((r, w - 1 - c))

    for r, c in border_points:
        if not self.thresholded[r][c]:
          continue
        # The (255 << 8) incantation means set mask value to 255 when filling. The | 8 means do an 8-neighbor fill.
        cv2.floodFill(self.thresholded, mask, (c, r), 255, flags=(255 << 8) | cv2.FLOODFILL_MASK_ONLY | 8)

    # because we do a dilate3 in super().clean_after_crop, we also need to do that here so the mask matches when we
    # subtract
    mask = dilate(mask, 3)

    return mask[1:-1, 1:-1]


def ngroupwise(n, iterable):
  # generalization of the "pairwise" recipe
  iterators = list(itertools.tee(iterable, n))
  for i in range(n):
    for j in range(i):
      next(iterators[i], None)

  return zip(*iterators)


def threshold(img, min_value=170, max_saturation=25):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  return cv2.inRange(hsv, (0, 0, min_value), (179, max_saturation, 255))


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


def dilate(img, n=3):
  kernel = np.ones((n, n), np.uint8)
  return cv2.dilate(img, kernel)


def erode(img, n=3):
  kernel = np.ones((n, n), np.uint8)
  return cv2.erode(img, kernel)


def sharpen(img):
  blurred = cv2.GaussianBlur(img, (3, 3), 0)
  return cv2.addWeighted(img, 2, blurred, -1, 0)


def remove_small_islands(img, min_pixels=2):
  mask = np.zeros(img.shape)
  im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  for contour in contours:
    if cv2.contourArea(contour) < min_pixels:
      cv2.fillPoly(mask, pts=contour, color=(255, 255, 255))
  return img - mask


def show_image(img):
  # compute the name of the object we're displaying
  var_name = None
  lcls = inspect.stack()[1][0].f_locals
  if 'self' in lcls:
    for k, v in lcls['self'].__dict__.items():
      if id(img) == id(v):
        var_name = 'self.' + k
        break

  if var_name is None:
    for name in lcls:
      if name == '_':
        continue
      if id(img) == id(lcls[name]):
        var_name = name
        break

  if var_name is None:
    var_name = '(unknown image)'

  # resize image
  scale_factor = 4
  img = cv2.resize(img, (0, 0), None, scale_factor, scale_factor, cv2.INTER_NEAREST)

  cv2.imshow(var_name, img)
  while True:
    key = cv2.waitKey(0)
    if key == ord('q'):
      raise Exception('quitting')
    if ord(' ') <= key <= ord('~'):
      break
  cv2.destroyAllWindows()


def show_unprocessed_processed(unp, p):
  cv2.imshow('unprocessed', unp)
  cv2.imshow('processed', p)
  while True:
    k = cv2.waitKey(100)
    if k == ord('q'):
      raise Exception('quitting')
    elif k == ord('j'):
      break

  cv2.destroyAllWindows()


def pad_string(s, l):
  chars_taken = len(s)
  for c in s:
    if unicodedata.east_asian_width(c) == 'W':
      chars_taken += 1

  return s + ' ' * (l - chars_taken)


def get_all_test_frames():
  rv = []
  for dirpath, dirnames, filenames in os.walk('test_frames'):
    if 'unprocessed' in dirpath.split('/'):
      continue
    for filename in filenames:
      rv.append(os.path.join(dirpath, filename))

  return sorted(rv)


def test_all(model_class):
  passes = 0
  cases = 0
  for fname in get_all_test_frames():
    passes += test_case(model_class, fname)
    cases += 1

  print('==== passed {} / {} tests ({} %) ===='.format(
    passes, cases, int(round(passes / cases * 100))))


def test_case(model_class, fname, debug=False):
  img = cv2.imread(fname)
  expected_text = fname.split('__')[1][:-4]
  model = model_class(debug=debug)
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


MODELS = {
  'e0': E0,
  'e1': E1,
  'e2': E2,
  'e3': E3,
}


if __name__ == '__main__' and not hasattr(sys, 'ps1'):
  main(parser.parse_args())
