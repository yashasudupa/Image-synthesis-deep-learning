import numpy as np
from sklearn.feature_extraction import _stop_words
from sklearn import preprocessing
#from googletrans import Translator, constants
from pprint import pprint
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import csv

from PIL import Image  # from Pillow
import pytesseract
from pytesseract import image_to_string

from pdf2image import convert_from_path

# assign directory
directory = 'Images'

# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

filepath = r'/home/yashas123/Desktop/NLP_for_Form_Recognition/Multi-text-classification/'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

# iterate over files in
# that directory

list_of_texts = []

# Images to text
def Image_to_text():

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a little
        if os.path.isfile(f):
            # Store Pdf with convert_from_path function
            images = convert_from_path(f)

            for i in range(len(images)):
                # Save pages as images in the pdf
                images[i].save('page'+ filename + str(i) +'.jpg', 'JPEG')

                img = Image.open(filepath + 'page'+ filename + str(i) +'.jpg')

                image = cv2.imread(filepath + 'page'+ filename + str(i) +'.jpg')
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                thresh = 255 - cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

                # Blur and perform text extraction
                thresh = cv2.GaussianBlur(thresh, (3,3), 0)
                data = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6')
                print(data)

                # Extract table from the Image
                Extract_Table_from_Image(filepath + 'page'+ filename + str(i) +'.jpg')

                #text = pytesseract.image_to_string(img) # Timeout after 2 seconds

                # Displaying the extracted text
                #print(text[:-1])


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


# Images to text
def Extract_Table_from_Image(filepath=filepath):
    
    #read your file
    img = cv2.imread(filepath,0)
    print(img.shape)

    #thresholding the image to a binary image
    thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY |cv2.THRESH_OTSU)
    
    #inverting the image 
    #img_bin = 255-img_bin'
    cv2.imwrite('/home/yashas123/Desktop/NLP_for_Form_Recognition/Multi-text-classification/cv_inverted.png',img_bin)
    
    #Plotting the image to see the output
    plotting = plt.imshow(img_bin,cmap='gray')
    #plt.show()

    #Detection of Horizontal Lines
    # Length(width) of kernel as 100th of total width
    kernel_len = np.array(img).shape[1]//100
    # Defining a vertical kernel to detect all vertical lines of image 
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    #Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
    cv2.imwrite("/home/yashas123/Desktop/NLP_for_Form_Recognition/Multi-text-classification/vertical.jpg",vertical_lines)
    #Plot the generated image
    plotting = plt.imshow(image_1,cmap='gray')
    #plt.show()

    #Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
    cv2.imwrite("/home/yashas123/Desktop/NLP_for_Form_Recognition/Multi-text-classification/horizontal.jpg",horizontal_lines)
    #Plot the generated image
    plotting = plt.imshow(image_2,cmap='gray')
    #plt.show()

    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    #Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite("/home/yashas123/Desktop/NLP_for_Form_Recognition/Multi-text-classification/img_vh.jpg", img_vh)
    bitxor = cv2.bitwise_xor(img,img_vh)
    bitnot = cv2.bitwise_not(bitxor)
    #Plotting the generated image
    plotting = plt.imshow(bitnot,cmap='gray')
    #plt.show()

    # Detect contours for following box detection
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort all the contours by top to bottom.
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

    #Creating a list of heights for all detected boxes
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]

    #Get mean of heights
    mean = np.mean(heights)

    #Create list box to store all boxes in  
    box = []
    # Get position (x,y), width and height for every contour and show the contour on image
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w<1000 and h<500):
            image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            box.append([x,y,w,h])
    plotting = plt.imshow(image,cmap='gray')
    #plt.show()

    #Creating two lists to define row and column in which cell is located
    row=[]
    column=[]
    j=0
    #Sorting the boxes to their respective row and column
    for i in range(len(box)):
        if(i==0):
            column.append(box[i])
            previous=box[i]
        else:
            if(box[i][1]<=previous[1]+mean/2):
                column.append(box[i])
                previous=box[i]
                if(i==len(box)-1):
                    row.append(column)
            else:
                row.append(column)
                column=[]
                previous = box[i]
                column.append(box[i])

    #calculating maximum number of cells
    countcol = 0
    for i in range(len(row)):
        countcol = len(row[i])
        if countcol > countcol:
            countcol = countcol

    #Retrieving the center of each column
    center = [int(row[i][j][0]+row[i][j][2]/2) for j in range(len(row[i])) if row[0]]
    center=np.array(center)
    center.sort()

    #Regarding the distance to the columns center, the boxes are arranged in respective order
    finalboxes = []
    for i in range(len(row)):
        lis=[]
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)

    #from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
    outer=[]
    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            inner=''
            if(len(finalboxes[i][j])==0):
                outer.append(' ')
            else:
                for k in range(len(finalboxes[i][j])):
                    y,x,w,h = finalboxes[i][j][k][0],finalboxes[i][j][k][1], finalboxes[i][j][k][2],finalboxes[i][j][k][3]
                    finalimg = bitnot[x:x+h, y:y+w]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    border = cv2.copyMakeBorder(finalimg,2,2,2,2,   cv2.BORDER_CONSTANT,value=[255,255])
                    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    dilation = cv2.dilate(resizing, kernel,iterations=1)
                    erosion = cv2.erode(dilation, kernel,iterations=1)
            
                    out = pytesseract.image_to_string(erosion)
                    if(len(out)==0):
                        out = pytesseract.image_to_string(erosion, config='--psm 3')
                    inner = inner +" "+ out
                outer.append(inner)
    print(outer)

#def text_to_bag_of_words():

    #Categorize sentences based on the seperation
    # Remove unnecessary words
    # pre-processing


    # Use English Dictinoary
    # Seperate texts based on the seperation
    # Remove stopping words and unwanted words
    # Nouns and pronouns must be kept
    # Extra fields must be added if necessary


#def process_text_for_synonymys():

#def 

if __name__ == '__main__':
    Image_to_text()
        