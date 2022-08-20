import cv2 
import pytesseract
from pytesseract import image_to_string
import numpy

# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

filepath = r'/home/yashas123/Desktop/NLP_for_Form_Recognition/Multi-text-classification/'

img = cv2.imread('pageTSR.jpg')

# Adding custom options
custom_config = r'--oem 3 --psm 6'
pytesseract.image_to_string(img, config=custom_config)