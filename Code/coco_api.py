import os
import numpy as np
import cvbase as cvb
import cv2
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
import pdb
import copy


def polys_to_mask(polygons, height, width):
    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle)
    return mask

def obj_mask(i_m):
    i_m = i_m.astype(np.uint8) * 255
    i_m = np.stack((i_m, i_m, i_m), axis=2)
    return i_m

def occ_mask(a_m, i_m):
    a_m = a_m.astype(np.uint8) * 255
    i_m = i_m.astype(np.uint8) * 255
    a_m = np.stack((a_m, a_m, a_m), axis=2)
    i_m = np.stack((i_m, i_m, i_m), axis=2)

    occ_m = a_m - i_m
    occ_m = np.maximum(occ_m, 0)
    return occ_m

def vis_mask(img, a_m, i_m):
    cv2.namedWindow("a_i_m")
    a_i = copy.deepcopy(img)
    i_i = copy.deepcopy(img)
    occ_i = copy.deepcopy(img)

    a_m = a_m.astype(np.uint8) * 255
    i_m = i_m.astype(np.uint8) * 255
    a_m = np.stack((a_m, a_m, a_m), axis=2)
    i_m = np.stack((i_m, i_m, i_m), axis=2)
    occ_m = a_m - i_m
    occ_m = np.maximum(occ_m, 0)

    a_m_w = cv2.addWeighted(a_i, 0.3, a_m, 0.7, 0)
    i_m_w = cv2.addWeighted(i_i, 0.3, i_m, 0.7, 0)
    occ_m_w = cv2.addWeighted(occ_i, 0.3, occ_m, 0.7, 0)
    a_i_m = np.concatenate((a_m_w, i_m_w, occ_m_w), axis=0)

    cv2.imshow("a_i_m", a_i_m)
    return cv2.waitKey(10000)

def make_json_dict(imgs, anns):
    imgs_dict = {}
    anns_dict = {}
    for ann in anns:
        image_id = ann["image_id"]
        if not image_id in anns_dict:
            anns_dict[image_id] = []
            anns_dict[image_id].append(ann)
        else:
            anns_dict[image_id].append(ann)

    for img in imgs:
        image_id = img['id']
        imgs_dict[image_id] = img['file_name']

    return imgs_dict, anns_dict


is_train = False
img_num = 100
data_dir = '../kins_cuts/'

if is_train:
    base_img_path = "../data_object_image_2/training/image_2/"
    base_ann_path = "./update_train_2020.json"
else:
    base_img_path = "../data_object_image_2/testing/image_2/"
    base_ann_path = "./update_test_2020.json"

coco = COCO(base_ann_path)
cats = coco.loadCats(coco.getCatIds())
cats_nms = [cat['name'] for cat in cats]
# print('cats: \n{}\n'.format(' '.join(cats_nms)))
catIds = coco.getCatIds(catNms=cats_nms)
catDic = dict(zip(catIds, cats_nms))

anns = cvb.load(base_ann_path)
imgs_info = anns['images']
anns_info = anns["annotations"]

imgs_dict, anns_dict = make_json_dict(imgs_info, anns_info)
img_file_dict = dict(zip(cats_nms, [[] for i in range(len(cats_nms))] ))

for img_id in anns_dict.keys():
    img_name = imgs_dict[img_id]

    img_path = os.path.join(base_img_path, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    height, width, _ = img.shape
    anns = anns_dict[img_id]

    for ann in anns:
        # load amodal mask & inmodal mask
        a_mask = polys_to_mask(ann["a_segm"], height, width)
        i_mask = polys_to_mask(ann["i_segm"], height, width)
        x, y, w, h = ann["a_bbox"]

        obj_m = obj_mask(i_mask)
        obj_m = obj_m[y: y+h, x: x+w]
        obj_dir = data_dir + "{}_occludee_mask/".format(catDic[ann["category_id"]])
        if not os.path.exists(obj_dir):
            os.makedirs(obj_dir)
        cv2.imwrite(obj_dir + "{:0>5d}_{:0>6d}.png".format(img_id, ann["id"]), obj_m)

        occ_m = occ_mask(a_mask, i_mask)
        occ_m = occ_m[y: y+h, x: x+w]
        occ_dir = data_dir + "{}_mask/".format(catDic[ann["category_id"]])
        if not os.path.exists(occ_dir):
            os.makedirs(occ_dir)
        cv2.imwrite(occ_dir + "{:0>5d}_{:0>6d}.png".format(img_id, ann["id"]), occ_m)

        # rt_key = vis_mask(img, a_mask, i_mask)
        # # press 'q' or 'esc' to stop displaying images
        # if rt_key == ord('q') or rt_key == 27:
        #     cv2.destroyAllWindows()
        #     exit()

        # # generate signel object image
        # cropped = img[y: y+h, x: x+w]
        #
        # height, width, _ = cropped.shape
        # # cv2.imshow('cropped', cropped)
        # # cv2.waitKey(2000)
        #
        # crop_dir = data_dir + "{}/".format(catDic[ann["category_id"]])
        # if not os.path.exists(crop_dir):
        #     os.makedirs(crop_dir)
        #
        # cv2.imwrite(crop_dir + "{:0>5d}_{:0>6d}.png".format(img_id, ann["id"]), cropped)
        # img_file_dict[catDic[ann["category_id"]]].append("{:0>5d}_{:0>6d}\n".format(img_id, ann["id"]))

    if img_id >= img_num:
        break

# for cat in cats_nms:
#     f = open(data_dir + cat + "_occ.txt", "w+")
#     for img_file in img_file_dict[cat]:
#         f.write(img_file)
#     f.close()