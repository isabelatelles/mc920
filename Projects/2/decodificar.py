from argparse import ArgumentParser
from imageio import imread, imwrite
from matplotlib.pyplot import imshow, show
from numpy import ndarray


# Defines the required arguments to run the program
def define_arguments():
    parser = ArgumentParser()
    parser.add_argument('output_image', metavar='OUTPUT_IMAGE_PATH', nargs='?',
                        help='path to the image file with embedded message')
    parser.add_argument('bit_plane', metavar='BIT_PLANE', type=int, choices=range(0, 3), nargs='?',
                        help='bit plane indicating in which layer of the image the message was hidden')
    parser.add_argument('output_text', metavar='OUTPUT_TEXT_PATH', nargs='?',
                        help='path to the text file with the recovered message')
    args = parser.parse_args()

    return args.output_image, args.bit_plane, args.output_text


# Decodes a message into an image and
# Creates bit planes 0, 1, 2 and 7 in order to visualize the modification into the bit plane
def decode_image(image_path, image, bit_plane):
    height, width, channel = image.shape

    name, extension = image_path.split('.')
    # Initializes bit planes
    bit_plane_zero = ndarray(image.shape, dtype='uint8')
    bit_plane_one = ndarray(image.shape, dtype='uint8')
    bit_plane_two = ndarray(image.shape, dtype='uint8')
    bit_plane_seven = ndarray(image.shape, dtype='uint8')

    # Binary code of ascii
    binary_code = ''
    # Iterates byte by byte (8 bits)
    i = 0
    # String with text of message hidden
    text = ''
    for h in range(height):
        for w in range(width):
            for c in range(channel):
                # Constructs bit planes 0, 1, 2 and 7
                bit_plane_zero[h][w][c] = ((image[h][w][c] >> 0) % 2) * 255
                bit_plane_one[h][w][c] = ((image[h][w][c] >> 1) % 2) * 255
                bit_plane_two[h][w][c] = ((image[h][w][c] >> 2) % 2) * 255
                bit_plane_seven[h][w][c] = ((image[h][w][c] >> 7) % 2) << 7

                i += 1
                bit = ((image[h][w][c] >> bit_plane) % 2)
                binary_code += str(bit)
                if i == 8:
                    char = chr(int(binary_code, 2))
                    text += char
                    i = 0
                    binary_code = ''

    show_and_save_image(name + '_0.' + extension, bit_plane_zero)
    show_and_save_image(name + '_1.' + extension, bit_plane_one)
    show_and_save_image(name + '_2.' + extension, bit_plane_two)
    show_and_save_image(name + '_7.' + extension, bit_plane_seven)

    return text


def write_text(text, text_path):
    with open(text_path, 'w') as f:
        f.write(text)


def show_and_save_image(image_path, image):
    imshow(image)
    show()
    imwrite(image_path, image)


def main():
    output_image, bit_plane, output_text = define_arguments()
    img = imread(output_image).astype(dtype='uint8')
    text = decode_image(output_image, img, bit_plane)
    write_text(text, output_text)


if __name__ == "__main__":
    main()
