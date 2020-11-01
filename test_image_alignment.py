import unittest
from ImageAlignment import ImageAlignment
import cv2

class MyTestCase(unittest.TestCase):
    def test_load_image_as_grayscale(self):
        test_img_path = "Tests/18-04-2018.jpg"
        actual_output = ImageAlignment().load_image_as_grayscale(test_img_path)
        expected_output = cv2.imread("Tests/grayscale_output.png", cv2.IMREAD_GRAYSCALE)

        # All pixels should match identically
        self.assertTrue((actual_output==expected_output).all())

    def test_align_image(self):
        test_img_path = "downscaled.png"
        test_form_path = "Templates/FitnessFormFront.jpg"

        input_img = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
        input_template = cv2.imread(test_form_path, cv2.IMREAD_COLOR)

        output_img = ImageAlignment().align_images(input_img, input_template)
        cv2.imwrite("aligned2.png", output_img)




if __name__ == '__main__':
    unittest.main()
