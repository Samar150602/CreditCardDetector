import cv2
import numpy as np
from imutils import contours
import imutils

class CreditCardOCR:
    """
    Class for performing Optical Character Recognition (OCR) on credit card images using template matching.

    Attributes:
    - reference_image_path (str): The path to the reference image containing digit templates.
    - min_roi_size (int): The minimum size for a Region of Interest (ROI) to be considered (default is 100).
    - digits (dict): A dictionary to store digit templates for matching during OCR.
    """

    def __init__(self, reference_image_path, min_roi_size=100):
        """
        Initializes the CreditCardOCR object.

        Parameters:
        - reference_image_path (str): The path to the reference image containing digit templates.
        - min_roi_size (int): The minimum size for a Region of Interest (ROI) to be considered (default is 100).
        """
        self.min_roi_size = min_roi_size
        self.digits = {}
        self.load_reference_image(reference_image_path)

    def load_reference_image(self, reference_image_path):
        """
        Loads the reference image, preprocesses it, and extracts digit templates.

        Parameters:
        - reference_image_path (str): The path to the reference image containing digit templates.
        """
        ref = cv2.imread(reference_image_path)
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        ref = cv2.threshold(ref, 120, 255, cv2.THRESH_BINARY_INV)[1]

        refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refCnts = imutils.grab_contours(refCnts)
        refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]

        for (i, c) in enumerate(refCnts):
            (x, y, w, h) = cv2.boundingRect(c)
            roi = ref[y:y + h, x:x + w]
            if (len(roi) * len(roi[0])) < self.min_roi_size:
                continue
            roi = cv2.resize(roi, (57, 88))
            self.digits[i] = roi

    def process_image(self, image_path):
        """
        Processes a credit card image using OCR to extract and recognize the credit card number.

        Parameters:
        - image_path (str or numpy.ndarray): The path to the credit card image or the image array itself.

        Returns:
        - bool: True if OCR successfully extracts and recognizes the credit card number, False otherwise.
        """
        FIRST_NUMBER = {
            "3": "American Express",
            "4": "Visa",
            "5": "MasterCard",
            "6": "Discover Card"
        }

        # Load the image
        if type(image_path) == str:
            image = cv2.imread(image_path)
        else:
            image = image_path
        image = imutils.resize(image, width=600)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply image processing steps for OCR
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
        gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
        gradX = gradX.astype("uint8")

        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        kernel = np.ones((3, 3), np.uint8)

        thresh = cv2.dilate(thresh, kernel, iterations=10)
        thresh = cv2.erode(thresh, kernel, iterations=1)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        locs = []

        print('cnts', len(cnts))

        # Loop through contours and filter based on aspect ratio and size
        for (i, c) in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)

            color = (0, 255, 0)  # Green color in BGR
            thickness = 1

            temp_image = image.copy()

            # Display the image with the rectangle
            if 180 < y < 220:
                if w > 150:
                    w = 110
                if h > 60:
                    h = 45
                ar = w / float(h)
                cv2.rectangle(temp_image, (x, y), (x + w, y + h), color, thickness)
                print('ar', ar, 'width', w, 'height', h, 'x', x, 'y', y)

                if ar > 1.75 and ar < 3:
                    if (w > 90 and w < 140) and (h > 35 and h < 55):
                        cv2.imshow('Image with Rectangle', temp_image)
                        cv2.waitKey(2000)
                        print('reached')
                        locs.append((x, y, w, h))

        locs = sorted(locs, key=lambda x: x[0])
        print('locs', len(locs))
        output = []

        # Loop through the detected regions and perform digit recognition
        for (i, (gX, gY, gW, gH)) in enumerate(locs):
            groupOutput = []
            group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
            group = cv2.threshold(group, 100, 255, cv2.THRESH_BINARY)[1]
            group = cv2.dilate(group, kernel, iterations=1)
            cv2.imshow('numb', group)
            cv2.waitKey(2000)

            digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            digitCnts = imutils.grab_contours(digitCnts)
            digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

            for c in digitCnts:
                (x, y, w, h) = cv2.boundingRect(c)
                roi = group[y:y + h, x:x + w]

                if (len(roi) * len(roi[0])) < self.min_roi_size:
                    continue

                roi = cv2.resize(roi, (57, 88))
                scores = []

                for (digit, digitROI) in self.digits.items():
                    result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                    (_, score, _, _) = cv2.minMaxLoc(result)
                    scores.append(score)

                groupOutput.append(str(np.argmax(scores)))

            cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
            cv2.putText(image, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            output.extend(groupOutput)

        # Display the final result if a valid credit card number is detected
        if len(output) == 16:
            print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
            print("Credit Card #: {}".format("".join(output)))
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            return True
        else:
            cv2.imshow("Image", image)
            cv2.waitKey(2000)
            return False
