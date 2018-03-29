# To run the code, type:
#    python3.5 trabalho_1.py image_file
# Example:
#    python3.5 trabalho_1.py objetos1.png

from argparse import ArgumentParser
import matplotlib.pyplot as plt
from numpy import shape, empty
import cv2
from skimage.measure import label, regionprops


# Defines the required argument, which is the path to the image file
# Returns the image file path received as argument
def define_image_argument():
    parser = ArgumentParser()
    parser.add_argument('image', metavar='IMAGE_PATH', nargs='?', help='path to the image file')
    args = parser.parse_args()

    return args.image


# Reads and returns the color image as a 3D matrix
def read_image(image_path):
    return cv2.imread(image_path)


# Shows image and closes window if any key is pressed
def show_image(window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Saves image in current folder
def save_image(image_path, image):
    cv2.imwrite(image_path, image)


# Creates an image with the contours of the objects
# Returns a tuple of an image with the objects contours and an binary image obtained after threshold
def create_img_contours(image, gray_image):
    ret, thresh = cv2.threshold(gray_image, thresh=190, maxval=255, type=cv2.THRESH_BINARY_INV)
    img, contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    white_image = empty([shape(image)[0], shape(image)[1], shape(image)[2]])
    white_image.fill(255)
    img_contours = cv2.drawContours(white_image, contours, contourIdx=-1, color=(0, 0, 255), thickness=2)

    return img_contours, thresh


# Extracts objects properties like perimeter, area and centroids from a binary image
# Returns a 1D numpy array with the area of each object
def extract_objects_properties(binary_image, gray_image, image_path):
    label_img, num_regions = label(binary_image, return_num=True)
    areas = empty([num_regions])
    plt.imshow(gray_image, cmap='gray')

    for i, region in enumerate(regionprops(label_img)):
        areas[i] = region.area
        print('region: {0}'.format(i), end='\t\t')
        print('perimeter: {0:.0f}'.format(region.perimeter), end='\t\t')
        print('area: {0:.0f}'.format(region.area), end='\n')
        plt.text(region.centroid[1], region.centroid[0], str(i), fontsize=10, color='white',
                 horizontalalignment='center', verticalalignment='center')
    print()
    plt.savefig('labeled_' + image_path)
    plt.show()

    return areas


# Prints the number of objects classified in each class - small, medium or big - according to their area
def print_classified_regions(classified_regions):
    print('number of small regions: {0:.0f}'.format(classified_regions[0]), end='\n')
    print('number of medium regions: {0:.0f}'.format(classified_regions[1]), end='\n')
    print('number of big regions: {0:.0f}'.format(classified_regions[2]), end='\n')


# Creates histogram of objects area
# Bins: 0 until 1500, 1500 until 3000, greater than or equal to 3000
# Returns an array of classified regions
def create_histogram(areas, image_path):
    plt.xlabel('Area')
    plt.ylabel('Number of Objects')
    plt.title('Histogram of Objects Area')
    max_area = max(areas)
    if max_area < 3000:
        max_area = 3000
    plt.xlim([0, max_area])
    classified_regions, bins, patches = plt.hist(areas, bins=[0, 1500, 3000, max_area], rwidth=0.5, facecolor='blue',
                                                 edgecolor='black')
    plt.savefig('histogram_' + image_path)
    plt.show()

    return classified_regions


def main():
    img_path = define_image_argument()
    img = read_image(img_path)

    show_image('original_image', img)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_image('gray_image', gray_img)
    save_image('gray_' + img_path, gray_img)

    img_contours, binary_img = create_img_contours(img, gray_img)
    show_image('img_contours', img_contours)
    save_image('contours_' + img_path, img_contours)

    areas = extract_objects_properties(binary_img, gray_img, img_path)

    classified_regions = create_histogram(areas, img_path)
    print_classified_regions(classified_regions)


if __name__ == "__main__":
    main()
