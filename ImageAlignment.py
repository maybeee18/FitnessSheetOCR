import cv2

class ImageAlignment:

    def load_image_as_grayscale(self, img_path):
        return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)