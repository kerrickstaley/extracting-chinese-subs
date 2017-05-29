import cv2

TEXT_TOP = 621
TEXT_BOTTOM = 684
TEXT_LEFT = 250
TEXT_RIGHT = 1030


img = cv2.imread('car_scene.png')
cropped = img[TEXT_TOP:TEXT_BOTTOM, TEXT_LEFT:TEXT_RIGHT]
cv2.imwrite('car_scene_cropped.png', cropped)
cv2.imshow('cropped', cropped)
cv2.waitKey(10000)
