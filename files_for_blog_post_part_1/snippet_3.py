white_region = cv2.inRange(cropped, (200, 200, 200), (255, 255, 255))
cv2.imwrite('car_scene_white_region.png', white_region)
cv2.imshow('white_region', white_region)
cv2.waitKey(10000)
