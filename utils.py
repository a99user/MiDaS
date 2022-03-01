"""Utils for monoDepth.
"""
import sys
import re
import numpy as np
import cv2
import torch
from sklearn.cluster import KMeans


def read_pfm(path):
    """Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale


def write_pfm(path, image, scale=1):
    """Write pfm file.

    Args:
        path (str): pathto file
        image (array): data
        scale (int, optional): Scale. Defaults to 1.
    """

    with open(path, "wb") as file:
        color = None

        if image.dtype.name != "float32":
            raise Exception("Image dtype must be float32.")

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif (
            len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
        ):  # greyscale
            color = False
        else:
            raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

        file.write("PF\n" if color else "Pf\n".encode())
        file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == "<" or endian == "=" and sys.byteorder == "little":
            scale = -scale

        file.write("%f\n".encode() % scale)

        image.tofile(file)


def read_image(path):
    """Read image and output RGB image (0-1).

    Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img[:720, :, :]

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        mask = np.zeros(img.shape[:2])
    elif img.shape[2] == 4:
        mask = img[:, :, 3]
        mask = np.where(mask == 255, np.ones_like(mask), np.zeros_like(mask))
        img = img[:, :, :3]
    else:
        mask = np.zeros(img.shape[:2])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img, mask


def resize_image(img):
    """Resize image and make it fit for network.

    Args:
        img (array): image

    Returns:
        tensor: data ready for network
    """
    height_orig = img.shape[0]
    width_orig = img.shape[1]

    if width_orig > height_orig:
        scale = width_orig / 384
    else:
        scale = height_orig / 384

    height = (np.ceil(height_orig / scale / 32) * 32).astype(int)
    width = (np.ceil(width_orig / scale / 32) * 32).astype(int)

    img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    img_resized = (
        torch.from_numpy(np.transpose(img_resized, (2, 0, 1))).contiguous().float()
    )
    img_resized = img_resized.unsqueeze(0)

    return img_resized


def resize_depth(depth, width, height):
    """Resize depth map and bring to CPU (numpy).

    Args:
        depth (tensor): depth
        width (int): image width
        height (int): image height

    Returns:
        array: processed depth
    """
    depth = torch.squeeze(depth[0, :, :, :]).to("cpu")

    depth_resized = cv2.resize(
        depth.numpy(), (width, height), interpolation=cv2.INTER_CUBIC
    )

    return depth_resized


def write_depth(path, depth, gt_mask, original_image, bits=1):
    """Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
    """
    # write_pfm(path + ".pfm", depth.astype(np.float32))

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.type)

    if bits == 1:
        out = out.astype("uint8")
    elif bits == 2:
        out = out.astype("uint16")

    # out = (out > 150) * out
    # print(out.shape)
    out = out * gt_mask
    out_depth = out.copy()

    out_filtered = out[out > 0]
    out_flatten = out_filtered.reshape(-1, 1)
    # print(out_flatten.shape)

    # kmeans = KMeans(n_clusters=2, random_state=0).fit(out_flatten)
    # pic2show = kmeans.cluster_centers_[kmeans.labels_]
    # out = pic2show.reshape(out.shape[0], out.shape[1]).astype(np.uint8)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(out_flatten)
    if kmeans.cluster_centers_[0, 0] - kmeans.cluster_centers_[1, 0] > 50:
        # mean_clusters_value = np.mean(kmeans.cluster_centers_)
        # out = np.where(out >= mean_clusters_value, out, np.zeros_like(out))
        bg_labels = out_flatten[kmeans.labels_ == 1]
        max_bg_value = np.max(bg_labels)
        out = np.where(out >= max_bg_value, out, np.zeros_like(out))

    new_gt_mask = np.where(out > 0, np.ones_like(out), out)

    first_row = np.concatenate((original_image, cv2.cvtColor(out_depth, cv2.COLOR_GRAY2BGR)), axis=1)
    second_row = np.concatenate(
        (original_image * new_gt_mask[..., None], cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)), axis=1)

    result = np.concatenate(
        (first_row, second_row), axis=0)

    # cv2.imwrite(path + ".png", out)
    cv2.imwrite(path + ".jpeg", result, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    return
