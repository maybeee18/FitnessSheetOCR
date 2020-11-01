import cv2
import numpy as np

class ImageAlignment:

    def load_image_as_grayscale(self, img_path):
        return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    def align_images(self, img, template_img):
        # Convert images to grayscale
        im1_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find size of image1
        sz = template_img.shape

        # Define the motion model
        warp_mode = cv2.MOTION_HOMOGRAPHY

        # Define the warp matrix
        warp_matrix = np.eye(3, 3, dtype=np.float32)

        # Define the number of iterations
        number_of_iterations = 500

        # Define correllation coefficient threshold
        # Specify the threshold of the increment in the correlation coefficient between two iterations
        termination_eps = 1e-10

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

        # Use warpPerspective for Homography
        im_aligned = cv2.warpPerspective(img, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        return im_aligned