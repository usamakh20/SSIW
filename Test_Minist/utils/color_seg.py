import xml.etree.ElementTree as ET
import numpy as np
import cv2

# Define the size of the rectangle
rectangle_size = 0.1

# Create a blank image to draw the legend on
legend_height = 700
legend_width = 300

font_scale = 0.8
thickness = 1
font = cv2.FONT_HERSHEY_SIMPLEX


def make_palette(num_classes=256):
    """
    Inputs:
        num_classes: the number of classes
    Outputs:
        palette: the colormap as a k x 3 array of RGB colors
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for k in range(0, num_classes):
        label = k
        i = 0
        while label:
            palette[k, 0] |= (((label >> 0) & 1) << (7 - i))
            palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
            palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
            label >>= 3
            i += 1
    idx1 = np.arange(0, num_classes, 2)[::-1]
    idx2 = np.arange(1, num_classes, 2)
    idx = np.concatenate([idx1[:, None], idx2[:, None]], axis=1).flatten()
    palette = palette[idx]
    palette[num_classes - 1, :] = [255, 255, 255]
    return palette


PALETTE = make_palette(256)


def color_seg(seg, palette=None):
    if palette == None:
        color_out = PALETTE[seg.reshape(-1)].reshape(seg.shape + (3,))
    else:
        color_out = palette[seg.reshape(-1)].reshape(seg.shape + (3,))
    return color_out


def color_map_list(class_num):
    map1 = np.asarray([
        [0, 0, 0],
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
        [165, 42, 42],
        [0, 192, 0],
        [196, 196, 196],
        [190, 153, 153],
        [180, 165, 180],
        [102, 102, 156],
        [128, 64, 255],
        [140, 140, 200],
        [170, 170, 170],
        [250, 170, 160],
        [96, 96, 96],
        [230, 150, 140],
        [128, 64, 128],
        [110, 110, 110],
        [244, 35, 232],
        [150, 100, 100],
        [70, 70, 70],
        [150, 120, 90],
        [220, 20, 60],
        [255, 0, 0],
        [200, 128, 128],
        [64, 170, 64],
        [128, 64, 64],
        [70, 130, 180],
        [152, 251, 152],
        [107, 142, 35],
        [0, 170, 30],
        [255, 255, 128],
        [250, 0, 30],
        [220, 220, 220],
        [222, 40, 40],
        [100, 170, 30],
        [40, 40, 40],
        [33, 33, 33],
        [0, 0, 142],
        [210, 170, 100],
        [153, 153, 153],
        [128, 128, 128],
        [250, 170, 30],
        [192, 192, 192],
        [220, 220, 0],
        [119, 11, 32],
        [0, 80, 100],
        [149, 32, 32],
        [10, 59, 140],
        [160, 0, 142],
        [0, 60, 100],
        [240, 100, 100]
    ])
    idx1 = np.arange(0, map1.shape[0], 2)[::-1]
    idx2 = np.arange(1, map1.shape[0], 2)
    idx = np.concatenate([idx1[:, None], idx2[:, None]], axis=1).flatten()
    map1 = map1[idx]

    pa = np.ones((class_num, 3), dtype=np.uint8) * 255
    pa[:map1.shape[0], :] = map1
    return pa


def add_legend(image, entries):
    legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
    # Loop through the colors and draw rectangles and labels
    for i, (color, label) in enumerate(entries):
        # Calculate the position of the rectangle
        x1 = int(legend_width * 0.05)
        y1 = int(i * 50 + 0.05 * legend_height)
        x2 = int(legend_width * (0.05 + rectangle_size))
        y2 = int((i + 1) * 50 - 0.05 * legend_height)
        # Draw the rectangle
        cv2.rectangle(legend, (x1, y1), (x2, y2), color, -1)

        # Draw the label
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = int(legend_width * (0.05 + rectangle_size + 0.05))
        text_y = int(y1 + (y2 - y1 + text_size[1]) / 2)
        cv2.putText(legend, label, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    # Find the largest height between the two images
    max_height = max(image.shape[0], legend.shape[0])

    # Determine the amount of padding required for each image
    padding1 = (max_height - image.shape[0]) // 2
    padding2 = (max_height - legend.shape[0]) // 2

    # Create a blank image with the largest height and the combined width
    combined_width = image.shape[1] + legend.shape[1]
    result = np.zeros((max_height, combined_width, 3), dtype=np.uint8)

    # Paste the first image into the blank image with padding
    result[padding1:padding1 + image.shape[0], :image.shape[1]] = image

    # Paste the second image into the blank image with padding
    result[padding2:padding2 + legend.shape[0], image.shape[1]:] = legend

    return result


def calculate_accuracy(xml_file, predicted):
    with open(xml_file) as f:
        data = '<root>' + f.read() + '</root>'

    predicted[predicted == 255] = 0
    root = ET.fromstring(data)
    mask = np.zeros(predicted.shape, dtype=np.uint8)
    for obj in root.findall('object'):
        label = int(obj.find('label').text)
        points = obj.find('points')
        x = [int(float(pt.text) * mask.shape[0]) - 1 for pt in points.findall('x')]
        y = [int(float(pt.text) * mask.shape[1]) - 1 for pt in points.findall('y')]
        cv2.rectangle(mask, (y[0], x[0]), (y[1], x[1]), label, -1)

    correct = np.equal(predicted, mask).sum()
    accuracy = correct / (mask.shape[0]*mask.shape[1])

    return accuracy
