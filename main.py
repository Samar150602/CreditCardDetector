# Import your classes here (replace ClassName1 and ClassName2 with the actual class names)
import sys
from find_card_v2 import CreditCardContourFinder
from find_card_details import CreditCardOCR

def main():
    # Example arguments for CreditCardContourFinder
    mobile_ip_address_cc = '192.168.24.106'
    mobile_port_number_cc = '8080'

    # Create an instance of CreditCardContourFinder
    credit_card_finder = CreditCardContourFinder(mobile_ip_address_cc, mobile_port_number_cc)

    # Create an instance of SomeOtherClass
    credit_card_details = CreditCardOCR(reference_image_path="font.jpg")
    
        # Now you can use the instances as needed
    while True:
        try:
            # card_cropped_image = credit_card_finder.find_credit_card_contour()
            # cv2.imshow('img', card_cropped_image)
            # cv2.waitKey(0)
            # cv2.imwrite('output_image.jpg', card_cropped_image)
            found = credit_card_details.process_image('card_3.jpg')
            print('found', found)
            if found:
                sys.exit()
        except:
            pass

if __name__ == "__main__":
    main()
