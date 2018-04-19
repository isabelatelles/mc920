from argparse import ArgumentParser
from imageio import imread, imwrite
from matplotlib.pyplot import imshow, show


# Defines the required arguments to run the program
def define_arguments():
    parser = ArgumentParser()
    parser.add_argument('input_image', metavar='INPUT_IMAGE_PATH', nargs='?',
                        help='path to the image file in which the message will be embedded')
    parser.add_argument('input_text', metavar='INPUT_TEXT_PATH', nargs='?',
                        help='path to the text file with the message to be hidden')
    parser.add_argument('bit_plane', metavar='BIT_PLANE', type=int, choices=range(0, 3), nargs='?',
                        help='bit plane indicating in which layer of the image the message will be hidden')
    parser.add_argument('output_image', metavar='OUTPUT_IMAGE_PATH', nargs='?',
                        help='path to the image file with embedded message')
    args = parser.parse_args()

    return args.input_image, args.input_text, args.bit_plane, args.output_image


def read_and_show_image(image_path):
    image = imread(image_path).astype(dtype='uint8')
    imshow(image)
    show()

    return image


# Converts the message to binary code
def convert_text_to_binary(text_path):
    with open(text_path, 'r') as f:
        text = f.read()
    binary = ''.join(format(ord(char), '08b') for char in text)

    return binary


# Sets bit (0 or 1) into the bit plane (0, 1 or 2) of the bit array, represented by an integer
def set_bit(bit_array, bit_plane, bit):
    # Creates mask, an integer in which the only bit set is the bit plane given
    mask = 1 << bit_plane
    if bit:
        bit_array = bit_array | mask
    else:
        bit_array = bit_array & ~mask

    return bit_array


# Encodes a message into an image
def encode_image(image, bit_plane, binary_code):
    height, width, channel = image.shape

    i = 0
    encoded_image = image

    for h in range(height):
        for w in range(width):
            for c in range(channel):
                # Sets bits of message into the image
                if i < len(binary_code):
                    encoded_image[h][w][c] = set_bit(image[h][w][c], bit_plane, int(binary_code[i]))
                    i += 1
                else:
                    return encoded_image

    return encoded_image


def show_and_save_image(image_path, image):
    imshow(image)
    show()
    imwrite(image_path, image)


def main():
    input_image, input_text, bit_plane, output_image = define_arguments()
    img = read_and_show_image(input_image)

    binary_code = convert_text_to_binary(input_text)

    encoded_img = encode_image(img, bit_plane, binary_code)
    show_and_save_image(output_image, encoded_img)


if __name__ == "__main__":
    main()
