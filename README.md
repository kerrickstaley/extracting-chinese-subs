# extracting-chinese-subs
This repository contains code to extract Chinese hard subs from the TV series 他来了请闭眼 (*Love Me If You Dare*). For further information please see [this post on my blog](http://www.kerrickstaley.com/2017/05/29/extracting-chinese-subs-part-1).

To get started, install OpenCV, Tesseract, the `chi_sim` data pack for Tesseract, and PyOCR. The following commands will work on Arch Linux:

```
sudo pacman -S opencv python-numpy tesseract tesseract-data-chi_sim
sudo pip install pyocr
```

Then try running `./main.py --test-all` to test the extraction algorithm on all test cases. To run it on a video file, you'll need to track down a 1280x720 video of one of the 他来了请闭眼 episodes with white hard subs at the bottom, similar to the test frames.
