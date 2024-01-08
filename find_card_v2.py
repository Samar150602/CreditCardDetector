import cv2

class CreditCardContourFinder:
    """
    A class for finding and processing credit card contours in a video stream.

    Attributes:
    - mobile_ip_address (str): The IP address of the mobile device streaming the video.
    - mobile_port_number (int): The port number used for video streaming on the mobile device.
    - target_height (int): The desired height for resizing the video frames.
    """

    def __init__(self, mobile_ip_address, mobile_port_number, target_height=500):
        """
        Initializes the CreditCardContourFinder object.

        Parameters:
        - mobile_ip_address (str): The IP address of the mobile device streaming the video.
        - mobile_port_number (int): The port number used for video streaming on the mobile device.
        - target_height (int): The desired height for resizing the video frames (default is 500).
        """
        self.mobile_ip_address = mobile_ip_address
        self.mobile_port_number = mobile_port_number
        self.target_height = target_height

    def resize_image(self, image):
        """
        Resizes an image to the specified target height while maintaining the aspect ratio.

        Parameters:
        - image (numpy.ndarray): The input image to be resized.

        Returns:
        - numpy.ndarray: The resized image.
        """
        ratio = self.target_height / image.shape[0]
        resized_image = cv2.resize(image, (int(image.shape[1] * ratio), self.target_height))
        return resized_image

    def crop_and_display_contour(self, frame, contour):
        """
        Crops the region defined by the contour from the given frame.

        Parameters:
        - frame (numpy.ndarray): The input frame from which the region is cropped.
        - contour (numpy.ndarray): The contour defining the region to be cropped.

        Returns:
        - numpy.ndarray: The cropped region of the frame.
        """
        x, y, w, h = cv2.boundingRect(contour)
        card_region = frame[y:y + h, x:x + w]
        return card_region

    def find_credit_card_contour(self):
        """
        Captures a video stream from a mobile device, processes frames, and finds the contour of a credit card.

        Returns:
        - numpy.ndarray: The cropped region containing the credit card.
        """
        video_stream_url = f'http://{self.mobile_ip_address}:{self.mobile_port_number}/video'
        cap = cv2.VideoCapture(video_stream_url)

        while True:
            ret, frame = cap.read()
            resized_frame = self.resize_image(frame)
            gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            largest_contour = None
            largest_contour_area = 0

            for contour in contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    if w < h:
                        temp = w
                        w = h
                        h = temp
                    aspect_ratio = float(w) / h

                    if 1.4 <= aspect_ratio <= 2.5:
                        contour_area = cv2.contourArea(contour)

                        if contour_area > largest_contour_area:
                            largest_contour_area = contour_area
                            largest_contour = approx

            if largest_contour is not None:
                card_region = self.crop_and_display_contour(resized_frame, largest_contour)
                return card_region

        cap.release()
        cv2.destroyAllWindows()
