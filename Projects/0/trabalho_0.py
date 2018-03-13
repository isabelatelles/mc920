from argparse import ArgumentParser
import matplotlib.pyplot as plt
from numpy import rint
from cv2 import imread, imwrite


# Defines the required argument, which is the path to the image file
# Returns the image file path received as argument
def define_image_argument():
    parser = ArgumentParser()
    parser.add_argument('image', metavar='IMAGE_PATH', nargs='?', help='path to the image file')
    args = parser.parse_args()
    return args.image


# Reads and returns the image as a matrix
def read_image(image_path):
    return imread(image_path, 0)


# Plots histogram
def plot_histogram(image_matrix):
    plt.hist(image_matrix.ravel(), bins=256, range=[0, 256])
    plt.title('Intensity Histogram')
    plt.xlabel('Levels of intensity')
    plt.ylabel('Frequency')
    plt.savefig('histogram.png')
    plt.show()


# Prints image's statistics
def print_statistics(img):
    print('Width: ' + str(img.shape[1]))
    print('Height: ' + str(img.shape[0]))
    print('Minimum level of intensity: ' + str(img.min()))
    print('Maximum level of intensity: ' + str(img.max()))
    print('Average level of intensity: ' + str(round(img.mean(), 2)))
    print()


# Reverses the intensity levels to produce the equivalent of a negative image
def negative_transformation(img):
    negative = 255 - img
    imwrite('negative.png', negative)


# Converts the intensity level range of [0,255] to [120,180]
def linear_transformation(img):
    converted_range = rint(img * (60.0/255.0)) + 120
    imwrite('linear_transformation.png', converted_range)


def main():
    img_path = define_image_argument()
    img = read_image(img_path)
    plot_histogram(img)
    print_statistics(img)
    negative_transformation(img)
    linear_transformation(img)


if __name__ == "__main__":
    main()
