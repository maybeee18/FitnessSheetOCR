import unittest
from PIL import Image
from ImageAlignment import ImageAlignment
import cv2

class MyTestCase(unittest.TestCase):
    def test_load_image_as_grayscale(self):
        test_img_path = "Tests/18-04-2018.jpg"
        actual_output = ImageAlignment().load_image_as_grayscale(test_img_path)
        expected_output = cv2.imread("Tests/grayscale_output.png", cv2.IMREAD_GRAYSCALE)

        # All pixels should match identically
        self.assertTrue((actual_output==expected_output).all())



if __name__ == '__main__':
    unittest.main()
