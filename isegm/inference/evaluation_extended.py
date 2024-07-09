from time import time
import os
import numpy as np
import torch
import cv2
from isegm.inference import utils
from isegm.inference.clicker import Clicker
import json

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm
    from tqdm import tqdm


def evaluate_dataset(dataset, predictor, resize=None, save_folder=None, **kwargs):
    all_ious = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        image = sample.image
        gt_mask = sample.gt_mask

        if not resize == None:
            img_height, img_width, _ = image.shape
            target_width, target_height = resize

            # Check if the image dimensions are larger than the resize dimensions, if yes, resize
            if img_width > target_width or img_height > target_height:
                image = cv2.resize(image, dsize=resize)
                gt_mask = cv2.resize(gt_mask, resize, interpolation=cv2.INTER_NEAREST)

        # Get original image dimensions before resizing
        orig_img_height, orig_img_width, _ = sample.image.shape

        _, sample_ious, _ = evaluate_sample(image, gt_mask, predictor,
                                            sample_id=index, save_folder=save_folder,
                                            dataset=dataset, orig_img_height=orig_img_height,
                                            orig_img_width=orig_img_width, **kwargs)

        all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time


# def evaluate_sample(image, gt_mask, predictor, max_iou_thr,
#                     pred_thr=0.5, min_clicks=1, max_clicks=20,
#                     sample_id=None, callback=None, save_folder=None, dataset=None):
#     clicker = Clicker(gt_mask=gt_mask)
#     pred_mask = np.zeros_like(gt_mask)
#     ious_list = []
#
#     if save_folder and not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#
#     with torch.no_grad():
#         predictor.set_input_image(image)
#
#         for click_indx in range(max_clicks):
#             clicker.make_next_click(pred_mask)
#
#             if clicker.not_improving:
#                 if callback is not None:
#                     callback(image, gt_mask, pred_probs, sample_id, click_indx - 1, clicker.clicks_list,
#                              clicker.not_improving)
#                 break
#
#             pred_probs = predictor.get_prediction(clicker)
#             pred_mask = pred_probs > pred_thr
#
#             iou = utils.get_iou(gt_mask, pred_mask)
#             ious_list.append(iou)
#
#             mask_save = pred_mask.astype(np.uint8)
#             mask_save *= 255 // mask_save.max()
#
#
#             # Saving all masks for all clicks to folder
#             image_name = os.path.splitext(dataset.dataset_samples[sample_id])[0]
#             save_path = os.path.join(save_folder, f"{image_name}_click_{click_indx + 1}.png")
#             #print("folder", save_folder)
#             #print("path", save_path)
#             cv2.imwrite(save_path, mask_save)
#
#             #BOUNDING BOX HERE
#
#             if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
#                 if callback is not None:
#                     callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)
#                 break
#
#             if click_indx == max_clicks - 1:
#                 if callback is not None:
#                     callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)
#
#     return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs

def evaluate_sample(image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.9, min_clicks=1, max_clicks=20, orig_img_width=None, orig_img_height=None,
                    sample_id=None, callback=None, save_folder=None, dataset=None):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []
    pred_thr=0.2 #change value here for the script to take it

    if save_folder and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with torch.no_grad():
        predictor.set_input_image(image)

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)

            if clicker.not_improving:
                if callback is not None:
                    callback(image, gt_mask, pred_probs, sample_id, click_indx - 1, clicker.clicks_list,
                             clicker.not_improving)
                break

            pred_probs = predictor.get_prediction(clicker)
            #print(f"Min probability: {pred_probs.min()}")
            #print(f"Max probability: {pred_probs.max()}")
            #print(f"Mean probability: {pred_probs.mean()}")
            #print(f"Thresholded mask sum: {pred_mask.sum()}")
            #print(pred_thr)
            #print(max_iou_thr)
            pred_mask = pred_probs > pred_thr

            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)

            mask_save = pred_mask.astype(np.uint8)
            mask_save *= 255 // mask_save.max()

            # Saving all masks for all clicks to folder
            image_name = os.path.splitext(dataset.dataset_samples[sample_id])[0]
            save_path = os.path.join(save_folder, f"{image_name}_click_{click_indx + 1}.png")
            cv2.imwrite(save_path, mask_save)

            # BOUNDING BOX HERE
            contours, _ = cv2.findContours(mask_save, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Combine all contours into a single list of points
            all_contour_points = np.concatenate(contours)

            # Find the minimum bounding rectangle that contains all points
            rect = cv2.minAreaRect(all_contour_points)

            # Get the bounding box coordinates and size
            x, y, w, h = cv2.boundingRect(all_contour_points)

            # Adjust bounding box coordinates using the ratio
            left = int(x * (orig_img_width / image.shape[1]))
            top = int(y * (orig_img_height / image.shape[0]))
            width = int(w * (orig_img_width / image.shape[1]))
            height = int(h * (orig_img_height / image.shape[0]))

            bounding_box = {
                "height": int(height),
                "width": int(width),
                "left": int(left),
                "top": int(top)
            }

            json_data = {
                "asset": {
                    "format": "jpg",
                    "id": f"{image_name}_click_{click_indx + 1}",
                    "name": f"{image_name}_click_{click_indx + 1}",
                    "path": os.path.join("/data/ritm_interactive_segmentation/datasets/Avalanche_uibk_test_all/images/",
                                         f"{image_name}.jpg"),
                    "size": {
                        "width": int(orig_img_width),
                        "height": int(orig_img_height)
                    }
                },
                "boundingBox": [bounding_box]  # Use a list with a single bounding box
            }

            # Save JSON file
            json_path = os.path.join(save_folder, f"{image_name}_click_{click_indx + 1}.json")
            with open(json_path, "w") as json_file:
                json.dump(json_data, json_file, indent=4)

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                if callback is not None:
                    callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)
                break

            if click_indx == max_clicks - 1:
                if callback is not None:
                    callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

    return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs

