# -*- coding: utf-8 -*-
from PIL import Image

def resize(img):
  img.thumbnail((32, 32), Image.ANTIALIAS)

  #横幅、縦幅を抜き出す
  width, height = img.size
  #画像サイズを加工
  square_size = min(img.size)

  if width > height:
      top = 0
      bottom = square_size
      left = (width - square_size) / 2
      right = left + square_size
      box = (left, top, right, bottom)
  else:
      left = 0
      right = square_size
      top = (height - square_size) / 2
      bottom = top + square_size
      box = (left, top, right, bottom)

  img = img.crop(box)

  img_resized = img.resize([28, 28])
  return img_resized