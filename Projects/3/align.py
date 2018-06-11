from argparse import ArgumentParser
import cv2 as cv2
import pytesseract as ocr
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# Define the required arguments to run the program
def define_arguments():
    parser = ArgumentParser()
    parser.add_argument('input_image', metavar='INPUT_IMAGE_PATH', nargs='?',
                        help='path to the PNG image file before the alignment')
    parser.add_argument('mode', metavar='MODE', type=int, choices=range(2), nargs='?',
                        help='technique used to align the image, ' +
                             '0: based on Hough transform,' +
                             '1: based on horizontal projection')
    parser.add_argument('output_image', metavar='OUTPUT_IMAGE_PATH', nargs='?',
                        help='path to the PNG image file after the alignment')
    args = parser.parse_args()

    return args.input_image, args.mode, args.output_image


# Show image and close window if any key is pressed
def show_image(window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def hough_transform(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    show_image('Edges of Image', edges)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    angles = []

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)
    show_image('Detected Lines in Image', image)
    cv2.imwrite('detected_lines.png', image)

    return median_angle


def horizontal_projection(thresh_image):
    # Set the max value of threshold image to 0, in order to represent a non-object pixel
    # and 0 to 1, in order to represent an object pixel
    binary_image = np.empty(thresh_image.shape)
    binary_image[thresh_image == 0] = 1
    binary_image[thresh_image != 0] = 0

    # Calculate and plot the horizontal projection profile of the image before skew correction
    row_sum_before = np.sum(binary_image, axis=1)
    plot_horizontal_projection_profile(row_sum_before)

    # Find skew angle and rotate the plot the horizontal projection profile of the image after skew correction
    skew_angle, row_sum = find_skew_angle_and_row_sum(binary_image, -90, 90)
    plot_horizontal_projection_profile(row_sum)

    return skew_angle


def plot_horizontal_projection_profile(row_sum):
    plt.barh(y=np.arange(row_sum.shape[0]), width=row_sum, height=1, color='black')
    plt.show()


def find_skew_angle_and_row_sum(image, min_theta, max_theta):
    max_score = None
    skew_angle = None
    row_sum_of_rotated_image = None
    for theta in range(min_theta, max_theta):
        score, row_sum = find_score_and_row_sum(image.copy(), theta)
        if max_score is None or score > max_score:
            max_score = score
            skew_angle = theta
            row_sum_of_rotated_image = row_sum

    return skew_angle, row_sum_of_rotated_image


# Rotate image and return its score and row sum
def find_score_and_row_sum(image, angle):
    rotated = rotate_image(image, angle)
    row_sum = np.sum(rotated, axis=1)
    score = np.sum((row_sum[1:] - row_sum[:-1]) ** 2)

    return score, row_sum


def rotate_image(image, angle):
    rows, cols = image.shape

    (cX, cY) = (cols // 2, rows // 2)
    # Grab the rotation matrix and the sine and cosine
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Compute the new bounding dimensions of the image
    nW = int((rows * sin) + (cols * cos))
    nH = int((rows * cos) + (cols * sin))

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Perform the actual rotation
    rotated = cv2.warpAffine(image, M, (nW, nH))

    return rotated


def write_text(text, text_path):
    with open(text_path, 'w') as f:
        f.write(text)


def main():
    input_image, mode, output_image = define_arguments()
    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    show_image('Original Image', img)

    thresh_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    text_original_image = ocr.image_to_string(Image.fromarray(thresh_image))
    write_text(text_original_image, input_image.replace('.png', '.txt'))

    skew_angle = None
    if mode == 0:
        skew_angle = hough_transform(img.copy())
    elif mode == 1:
        skew_angle = horizontal_projection(thresh_image)

    print('Skew Angle: ' + str(skew_angle))

    skew_free_image = rotate_image(img.copy(), skew_angle)
    show_image('Skew Free Image', skew_free_image)
    cv2.imwrite(output_image, skew_free_image)

    skew_free_image = cv2.adaptiveThreshold(skew_free_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    text_skew_free_image = ocr.image_to_string(Image.fromarray(skew_free_image))
    write_text(text_skew_free_image, output_image.replace('.png', '.txt'))


if __name__ == "__main__":
    main()

