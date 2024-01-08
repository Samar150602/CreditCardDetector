import sys
import tkinter as tk
from tkinter import Button, Label, filedialog
from find_card_v2 import CreditCardContourFinder
from find_card_details import CreditCardOCR

class CreditCardGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Credit Card Detection GUI")

        # Create GUI components
        self.status_label = Label(root, text="Status: Waiting for card detection...", font=('Helvetica', 14))
        self.status_label.pack(pady=20)

        self.image_button = Button(root, text="Use Image", command=self.detect_from_image)
        self.image_button.pack(pady=10)

        self.camera_button = Button(root, text="Use Camera Feed", command=self.detect_from_camera)
        self.camera_button.pack(pady=10)

    def detect_from_image(self):
        # Prompt user to select an image file
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if file_path:
            self.detect_card(file_path)

    def detect_from_camera(self):
        # Example arguments for CreditCardContourFinder
        mobile_ip_address_cc = '192.168.24.106'
        mobile_port_number_cc = '8080'

        # Create an instance of CreditCardContourFinder
        credit_card_finder = CreditCardContourFinder(mobile_ip_address_cc, mobile_port_number_cc)

        # Example arguments for CreditCardOCR
        credit_card_details = CreditCardOCR(reference_image_path="font.jpg")

        # Perform card detection
        found = False
        try:
            while True:
                card_cropped_image = credit_card_finder.find_credit_card_contour()
                # cv2.imshow('img', card_cropped_image)
                # cv2.waitKey(0)
                # cv2.imwrite('output_image.jpg', card_cropped_image)
                found = credit_card_details.process_image(card_cropped_image)
                print('found', found)
                if found:
                    sys.exit()
        except Exception as e:
            print(f"Error during detection: {e}")

        # Update the status label based on the detection result
        if found:
            self.status_label.config(text="Status: Card Found!")
        else:
            self.status_label.config(text="Status: Card Not Found!")

    def detect_card(self, image_path):
        # Example arguments for CreditCardContourFinder
        mobile_ip_address_cc = '192.168.24.106'
        mobile_port_number_cc = '8080'

        # Create an instance of CreditCardContourFinder
        credit_card_finder = CreditCardContourFinder(mobile_ip_address_cc, mobile_port_number_cc)

        # Example arguments for CreditCardOCR
        credit_card_details = CreditCardOCR(reference_image_path="font.jpg")

        # Perform card detection
        found = False
        try:
            found = credit_card_details.process_image(image_path)
            print('found', found)
        except Exception as e:
            print(f"Error during detection: {e}")

        # Update the status label based on the detection result
        if found:
            self.status_label.config(text="Status: Card Found!")
            sys.exit()
        else:
            self.status_label.config(text="Status: Card Not Found!")

if __name__ == "__main__":
    root = tk.Tk()
    app = CreditCardGUI(root)
    root.mainloop()
