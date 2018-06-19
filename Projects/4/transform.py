from argparse import ArgumentParser
import cv2 as cv2
import numpy as np
from math import floor, ceil, cos, sin


# Define the required arguments to run the program
def define_arguments():
    parser = ArgumentParser()
    parser.add_argument('input_image', metavar='INPUT_IMAGE_PATH', nargs=1,
                        help='path to the PNG image file before the geometric transformation')
    parser.add_argument('output_image', metavar='OUTPUT_IMAGE_PATH', nargs=1,
                        help='path to the PNG image file after the geometric transformation')
    parser.add_argument('-a', metavar='ANGLE_OF_ROTATION', type=float, nargs='?',
                        help='angle of rotation measured in degrees counterclockwise')
    parser.add_argument('-s', metavar='SCALE_FACTOR', type=float, nargs='?', help='scale factor')
    parser.add_argument('-d', metavar='OUTPUT_IMAGE_DIMENSION', type=int, nargs=2,
                        help='output image dimension in pixels')
    parser.add_argument('interpolation_method', metavar='INTERPOLATION_METHOD', type=int, choices=range(4), nargs=1,
                        help='interpolation method to be used: ' +
                        '0: nearest-neighbor interpolation, ' +
                        '1: bilinear interpolation, ' +
                        '2: bicubic interpolation, ' +
                        '3: Lagranges interpolation')
    args = parser.parse_args()

    return args.input_image[0], args.output_image[0], args.a, args.s, args.d, args.interpolation_method[0], parser


# Show image and close window if any key is pressed
def show_image(window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Rounds a number to floor if its fraction is less than half, or to ceil otherwise
def round_half_up(number):
    if number - floor(number) < 0.5:
        return int(floor(number))
    return int(ceil(number))


# Perform nearest neighbor interpolation (round half up)
# indexes: rows'/columns' indexes of the input image in which each row/column of the output image will be mapped
def nearest_neighbor_interpolation(indexes):
    return np.where(indexes - (indexes // 1) < 0.5, indexes // 1, indexes // 1 + 1).astype(int)


# Perform bilinear interpolation
# row_indexes: rows' indexes of the input image in which each row of the output image will be mapped
# col_indexes: columns' indexes of the input image in which each column of the output image will be mapped
# image: input image
# geometric transformation: scaling or rotation
def bilinear_interpolation(row_indexes, col_indexes, image, geometric_transformation):
    height, width = image.shape

    # Calculate floor and ceil values of rows' and columns' indexes
    floor_row_indexes = np.floor(row_indexes).astype(int)
    ceil_row_indexes = np.clip(floor_row_indexes + 1, 0, height - 1)
    floor_col_indexes = np.floor(col_indexes).astype(int)
    ceil_col_indexes = np.clip(floor_col_indexes + 1, 0, width - 1)

    northwest_neighbors = None
    southwest_neighbors = None
    northeast_neighbors = None
    southeast_neighbors = None

    # Calculate intensity of each neighbor
    if geometric_transformation == 'scaling':
        northwest_neighbors = image[floor_row_indexes, :][:, floor_col_indexes]
        southwest_neighbors = image[floor_row_indexes, :][:, ceil_col_indexes]
        northeast_neighbors = image[ceil_row_indexes, :][:, floor_col_indexes]
        southeast_neighbors = image[ceil_row_indexes, :][:, ceil_col_indexes]
    elif geometric_transformation == 'rotation':
        northwest_neighbors = image[floor_row_indexes, floor_col_indexes]
        southwest_neighbors = image[floor_row_indexes, ceil_col_indexes]
        northeast_neighbors = image[ceil_row_indexes, floor_col_indexes]
        southeast_neighbors = image[ceil_row_indexes, ceil_col_indexes]

    # Calculate fractional part of rows' and columns' indexes
    dy = row_indexes - floor_row_indexes
    dx = col_indexes - floor_col_indexes
    if geometric_transformation == 'scaling':
        dy = dy.reshape(len(dy), 1)
        dx = dx.reshape(1, len(dx))

    new_image = ((1 - dx) * (1 - dy) * northwest_neighbors +
                 dx * (1 - dy) * northeast_neighbors +
                 (1 - dx) * dy * southwest_neighbors +
                 dx * dy * southeast_neighbors).astype(np.uint8)

    return new_image


# Function P(t) used in bicubic interpolation
def function_p(t):
    return np.where(t > 0, t, t - t)


# Function R(s) used in bicubic interpolation
def function_r(s):
    return ((1/6) * (np.power(function_p(s + 2), 3) -
            4 * np.power(function_p(s + 1), 3) +
            6 * np.power(function_p(s), 3) -
            4 * np.power(function_p(s - 1), 3)))


# Function L(n) used in Lagranges interpolation in order to scale the image
def function_l_scaling(n, dx, image, floor_row_indexes, floor_col_indexes):
    height, width = image.shape
    cols = np.clip(floor_col_indexes + n - 2, 0, width - 1)
    f0 = image[np.clip(floor_row_indexes - 1, 0, height - 1), :][:, cols]
    f1 = image[floor_row_indexes, :][:, cols]
    f2 = image[np.clip(floor_row_indexes + 1, 0, height - 1), :][:, cols]
    f3 = image[np.clip(floor_row_indexes + 2, 0, height - 1), :][:, cols]
    c0, c1, c2, c3 = calculate_lagrange_coefficients(dx)

    return c0 * f0 + c1 * f1 + c2 * f2 + c3 * f3


# Function L(n) used in Lagranges interpolation in order to rotate the image
def function_l_rotation(n, dx, image, floor_row_indexes, floor_col_indexes):
    height, width = image.shape
    cols = np.clip(floor_col_indexes + n - 2, 0, width - 1)
    f0 = image[np.clip(floor_row_indexes - 1, 0, height - 1), cols]
    f1 = image[floor_row_indexes, cols]
    f2 = image[np.clip(floor_row_indexes + 1, 0, height - 1), cols]
    f3 = image[np.clip(floor_row_indexes + 2, 0, height - 1), cols]
    c0, c1, c2, c3 = calculate_lagrange_coefficients(dx)

    return c0 * f0 + c1 * f1 + c2 * f2 + c3 * f3


# Calculate Lagranges coefficients
def calculate_lagrange_coefficients(d):
    return (1/6) * (-d) * (d - 1) * (d - 2),\
           (1/2) * (d + 1) * (d - 1) * (d - 2),\
           (1/2) * (-d) * (d + 1) * (d - 2),\
           (1/6) * d * (d + 1) * (d - 1)


# Perform Lagranges interpolation
# row_indexes: rows' indexes of the input image in which each row of the output image will be mapped
# col_indexes: columns' indexes of the input image in which each column of the output image will be mapped
# image: input image
# geometric transformation: scaling or rotation
def lagrange_interpolation(row_indexes, col_indexes, image, geometric_transformation):
    # Calculate floor values of rows' and columns' indexes
    floor_row_indexes = np.floor(row_indexes).astype(int)
    floor_col_indexes = np.floor(col_indexes).astype(int)

    # Calculate fractional part of rows' and columns' indexes
    dy = row_indexes - floor_row_indexes
    dx = col_indexes - floor_col_indexes
    if geometric_transformation == 'scaling':
        dy = dy.reshape(len(dy), 1)
        dx = dx.reshape(1, len(dx))

    c0, c1, c2, c3 = calculate_lagrange_coefficients(dy)

    new_image = None
    if geometric_transformation == 'scaling':
        new_image = (c0 * function_l_scaling(1, dx, image, floor_row_indexes, floor_col_indexes) +
                     c1 * function_l_scaling(2, dx, image, floor_row_indexes, floor_col_indexes) +
                     c2 * function_l_scaling(3, dx, image, floor_row_indexes, floor_col_indexes) +
                     c3 * function_l_scaling(4, dx, image, floor_row_indexes, floor_col_indexes)).astype(np.uint8)
    elif geometric_transformation == 'rotation':
        new_image = (c0 * function_l_rotation(1, dx, image, floor_row_indexes, floor_col_indexes) +
                     c1 * function_l_rotation(2, dx, image, floor_row_indexes, floor_col_indexes) +
                     c2 * function_l_rotation(3, dx, image, floor_row_indexes, floor_col_indexes) +
                     c3 * function_l_rotation(4, dx, image, floor_row_indexes, floor_col_indexes)).astype(np.uint8)

    return new_image


# Scale image given the scale factor or the dimension of the output image, and the interpolation method
def image_scaling(image, scale_factor, dimension, interpolation_method):
    height, width = image.shape

    y_scale_factor = None
    x_scale_factor = None
    if scale_factor is not None:
        y_scale_factor = round(scale_factor, 3)
        x_scale_factor = round(scale_factor, 3)
    elif dimension is not None:
        y_scale_factor = round(dimension[0] / height, 3)
        x_scale_factor = round(dimension[1] / width, 3)

    new_height = round_half_up(y_scale_factor * height)
    new_width = round_half_up(x_scale_factor * width)

    # Calculate float values of row's and columns' indexes
    row_indexes = np.clip(np.arange(new_height) / y_scale_factor, 0, height - 1)
    col_indexes = np.clip(np.arange(new_width) / x_scale_factor, 0, width - 1)

    new_image = None
    if interpolation_method == 0:
        round_row_indexes = nearest_neighbor_interpolation(row_indexes)
        round_col_indexes = nearest_neighbor_interpolation(col_indexes)

        # Intensity values are attributed to pixels of the new transformed image
        new_image = image[round_row_indexes, :][:, round_col_indexes]

    elif interpolation_method == 1:
        new_image = bilinear_interpolation(row_indexes, col_indexes, image, 'scaling')

    elif interpolation_method == 2:
        # Calculate floor values of rows' and columns' indexes
        floor_row_indexes = np.floor(row_indexes).astype(int)
        floor_col_indexes = np.floor(col_indexes).astype(int)

        # Calculate fractional part of rows' and columns' indexes
        dy = row_indexes - floor_row_indexes
        dx = col_indexes - floor_col_indexes
        dy = dy.reshape(len(dy), 1)
        dx = dx.reshape(1, len(dx))

        # Perform bicubic interpolation
        new_image = np.zeros((new_height, new_width))
        for m in range(-1, 3):
            for n in range(-1, 3):
                rows = np.clip(floor_row_indexes + m, 0, height - 1)
                cols = np.clip(floor_col_indexes + n, 0, width - 1)
                f = image[rows, :][:, cols]
                rx = function_r(m - dx)
                ry = function_r(dy - n)
                new_image += f * rx * ry
        new_image = new_image.astype(np.uint8)

    elif interpolation_method == 3:
        new_image = lagrange_interpolation(row_indexes, col_indexes, image, 'scaling')

    show_image('Scaled image', new_image)

    return new_image


# Rotate image given the angle of rotation and the interpolation method
def image_rotation(image, angle, interpolation_method):
    height, width = image.shape

    rad_angle = np.deg2rad(angle)

    y_center = height / 2
    x_center = width / 2

    sin_angle = sin(rad_angle)
    cos_angle = cos(rad_angle)

    rows = np.arange(height).reshape(height, 1)
    cols = np.arange(width).reshape(1, width)

    # Calculate float values of row's and columns' indexes
    row_indexes = (cols - x_center) * sin_angle + (rows - y_center) * cos_angle + y_center
    col_indexes = (cols - x_center) * cos_angle - (rows - y_center) * sin_angle + x_center

    # Image with white padding of thickness 1
    pad_image = np.pad(image, 1, 'constant', constant_values=255)

    # Map every index out of limit to 0, so the background can be white, since (0, 0) is 255 because of the padding
    np.place(row_indexes, col_indexes < 0, 0)
    np.place(row_indexes, col_indexes >= width, 0)
    np.place(col_indexes, row_indexes < 0, 0)
    np.place(col_indexes, row_indexes >= height, 0)

    np.place(row_indexes, row_indexes < 0, 0)
    np.place(row_indexes, row_indexes >= height, 0)
    np.place(col_indexes, col_indexes < 0, 0)
    np.place(col_indexes, col_indexes >= width, 0)

    new_image = None
    if interpolation_method == 0:
        round_row_indexes = nearest_neighbor_interpolation(row_indexes)
        round_col_indexes = nearest_neighbor_interpolation(col_indexes)

        # Intensity values are attributed to pixels of the new transformed image
        new_image = pad_image[round_row_indexes, round_col_indexes]
    elif interpolation_method == 1:
        new_image = bilinear_interpolation(row_indexes, col_indexes, pad_image, 'rotation')

    elif interpolation_method == 2:
        # Calculate floor values of rows' and columns' indexes
        floor_row_indexes = np.floor(row_indexes).astype(int)
        floor_col_indexes = np.floor(col_indexes).astype(int)

        # Calculate fractional part of rows' and columns' indexes
        dy = row_indexes - floor_row_indexes
        dx = col_indexes - floor_col_indexes

        # Perform bicubic interpolation
        new_image = np.zeros((height, width))
        for m in range(-1, 3):
            for n in range(-1, 3):
                rows = np.clip(floor_row_indexes + m, 0, height + 1)
                cols = np.clip(floor_col_indexes + n, 0, width + 1)
                f = pad_image[rows, cols]
                rx = function_r(m - dx)
                ry = function_r(dy - n)
                new_image += f * rx * ry
        new_image = new_image.astype(np.uint8)

    elif interpolation_method == 3:
        new_image = lagrange_interpolation(row_indexes, col_indexes, pad_image, 'rotation')

    show_image('Rotated image', new_image)

    return new_image


def main():
    np.set_printoptions(threshold=np.nan)
    input_image, output_image, angle, scale_factor, dimension, interpolation_method, parser = define_arguments()

    if angle is None and (scale_factor is not None or dimension is not None):
        if scale_factor is not None and dimension is not None:
            ArgumentParser.error(parser,
                                 message='You should provide a scale factor or the dimension of the output image')
        image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
        transformed_image = image_scaling(image, scale_factor, dimension, interpolation_method)
        cv2.imwrite(output_image, transformed_image)
    elif scale_factor is None and angle is not None:
        if dimension is not None:
            ArgumentParser.error(parser,
                                 message='The output image dimension is not an argument allowed to rotate the image')
        image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
        transformed_image = image_rotation(image, angle, interpolation_method)
        cv2.imwrite(output_image, transformed_image)
    else:
        ArgumentParser.error(parser,
                             message='An angle of rotation, a scale factor or the dimension of the output image is ' +
                                     'necessary to run the program')


if __name__ == "__main__":
    main()

