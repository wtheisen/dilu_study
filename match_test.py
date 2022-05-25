import cv2
import feature_extractor

matches = feature_extractor.draw_keypoint_matches('./apu_1.png', './apu_2.png')

cv2.imwrite('./matched_image.png', matches)
