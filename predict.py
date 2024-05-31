import os
import time
from copy import deepcopy

import torch
import cv2
from det.utils.plotting import Annotator, colors

from rec import SVTR_G
from det import YOLO_SL

import pdb

def main():

    if check_img_path(img_path):
        print(f"{img_path} not found")
        return 0
    
    if os.path.isdir(img_path):
        for file in os.listdir(img_path):
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                filename = os.path.join(img_path, file)
                _predict_img(filename,
                     det_model,
                     weight_det_model,
                     rec_model,
                     save_dir,
                     device)
                
    if os.path.isfile(img_path) and img_path.endswith((".png", ".jpg", ".jpeg")):
        _predict_img(img_path,
                     det_model,
                     weight_det_model,
                     rec_model,
                     save_dir,
                     device)

def check_img_path(img_path):
    """
    Check if the given image path exists.

    Parameters:
        img_path (str): The path to the image.

    Returns:
        bool: True if the image path does not exist, False otherwise.
    """
    return not os.path.exists(img_path)

def _predict_img(
    img_path: str,
    det_model: YOLO_SL,
    weight_det_model: YOLO_SL,
    rec_model: SVTR_G,
    save_dir: str,
    device: str,
    ) -> None:
    """Predict the image at the given path.

    Args:
        img_path (str): The path to the image to be predicted.
        det_model (YOLO): The YOLOv5 model for detection.
        weight_det_model (YOLO): The YOLOv5 model for weight detection.
        rec_model (REC_MODEL): The Recognition model.
        save_dir (str): The directory to save the output image.
        device (str): The device to run the model on.

    Returns:
        None
    """
    det_obj = 0 # Counter for number of objects
    det_txt = 0 # Counter for number of text
    start_time = time.time()
    result = det_model.predict(source=img_path, save=False, imgsz=640, iou=0.75)
    boxes = _get_cls_boxes(result[0])
    annotator = Annotator(deepcopy(result[0].orig_img), 2, example=result[0].names,)
    for box in boxes:
        p1, p2 = (int(box[2][0]), int(box[2][1])), (int(box[2][2]), int(box[2][3]))
        p1 = (max(0, int(box[2][0])), max(0, int(box[2][1])))
        p2 = (min(result[0].orig_img.shape[1], int(box[2][2])), min(result[0].orig_img.shape[0], int(box[2][3])))
        if box[0] == 'c' or box[0] == 't':  # Handle container code and type
            vertical = False    # Assume it's not vertical
            roi = result[0].orig_img[p1[1]:p2[1], p1[0]:p2[0]]
            height, width, _ = roi.shape
            aspect_ratio = width / height
            if aspect_ratio < 1:
                vertical = True # If the aspect ratio is less than 1, it's vertical
                roi = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
                text = rec_model.val([roi], device)

            elif (aspect_ratio >= 1 and aspect_ratio < 2.35) and box[0] == 'c': # If the aspect ratio is between 1 and 2.35, it's a multi-line container code
                height = height // 2
                sub_rois = [roi[:height, :], roi[height:, :]]
                text = rec_model.val(sub_rois, device)
                text = [(text[0][0] + text[1][0], (float(text[0][1]) + float(text[1][1])) / 2)]

            else:
                text = rec_model.val([roi], device)

            label = f"{text[0][0].upper()} {str(text[0][1])[:4]}"
            annotator.box_label(box[2], label, color=colors(int(box[3]), True), vertical=vertical)
            det_txt += 1
        elif box[0] == 'w': # Handle weight
            annotator.lw = 1    # Set the line width to 1 for weight infomation
            roi_width = p2[0] - p1[0]
            roi_height = p2[1] - p1[1]
            enlarge_width = int(roi_width * 0.1)    # Enlarge the ROI witdth by 10%
            enlarge_height = int(roi_height * 0.05)     # Enlarge the ROI height by 5%
            transformed_img = _transform_weight_img(p1, p2, enlarge_width, enlarge_height, result[0].orig_img)
            w_results = weight_det_model.predict(transformed_img, imgsz=640)
            w_boxes = w_results[0].boxes.xyxy.cpu().numpy()
            cropped_images = []
            if w_boxes.size > 0:
                for w_box in w_boxes:
                    x1, y1, x2, y2 = map(int, w_box)
                    cropped_img = transformed_img[y1:y2, x1:x2, :]
                    cropped_images.append(cropped_img)
                
                texts = rec_model.val(cropped_images, device)
                for idx, scaled_w_box in enumerate(w_boxes):
                    # Scale the box back to the original image
                    scaled_w_box = [
                        (box[2][0] - enlarge_width // 2) + scaled_w_box[0] / 2,
                        (box[2][1] - enlarge_height // 2) + scaled_w_box[1] / 2,
                        (box[2][0] - enlarge_width // 2) + scaled_w_box[2] / 2,
                        (box[2][1] - enlarge_height // 2) + scaled_w_box[3] / 2
                    ]
                    label = f"{texts[idx][0].upper()} {str(texts[idx][1])[:4]}"
                    annotator.box_label(scaled_w_box, label, color=colors(int(box[3]), True))
                    det_txt += 1
            annotator.lw = 2    # Set the line width back to 2
        else:
            # Hazard sign
            label = f"{box[0]} {box[1]}"
            annotator.box_label(box[2], label, color=colors(int(box[3]), True))
            det_obj += 1

    print(f"Detected Object: {det_obj}, Detected Text: {det_txt}, Time: {time.time() - start_time:.4f}s, Image Save: {save_dir}/{os.path.basename(result[0].path)}")
    cv2.imwrite(f"{save_dir}/{os.path.basename(result[0].path)}", annotator.result())   # Save the draw image


def _transform_weight_img(p1, p2, enlarge_width, enlarge_height, orig_img):
    """
    Transforms a weight image by enlarging a region of interest (ROI) defined by the given coordinates.
    
    Args:
        p1 (tuple): The coordinates of the first point defining the ROI.
        p2 (tuple): The coordinates of the second point defining the ROI.
        enlarge_width (int): The amount to enlarge the width of the ROI.
        enlarge_height (int): The amount to enlarge the height of the ROI.
        orig_img (numpy.ndarray): The original image.
        
    Returns:
        numpy.ndarray: The transformed weight image.
    """
    p1_enlarged = (max(0, p1[0] - enlarge_width // 2), max(0, p1[1] - enlarge_height // 2))
    p2_enlarged = (min(orig_img.shape[1], p2[0] + enlarge_width // 2), min(orig_img.shape[0], p2[1] + enlarge_height // 2))
    roi_enlarged = orig_img[p1_enlarged[1]:p2_enlarged[1], p1_enlarged[0]:p2_enlarged[0]]
    w_img = cv2.cvtColor(roi_enlarged, cv2.COLOR_RGB2BGR)
    return _enhance_image(w_img)


def _enhance_image(image):
    """
    Enhances an image by resizing it to a larger resolution and applying Gaussian blur to remove noise.

    Parameters:
        image (numpy.ndarray): The input image to be enhanced.

    Returns:
        numpy.ndarray: The enhanced image.

    """
    height, width = image.shape[:2]
    resized_image = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
    blurred_image = cv2.GaussianBlur(resized_image, (3, 3), 0)

    return blurred_image

def _preprocess_config(det_model_1, det_model_2, rec_model):
    """
    Preprocesses the configuration for the given detection and recognition models.

    Args:
        det_model_1 (str): The name of the first detection model.
        det_model_2 (str): The name of the second detection model.
        rec_model (str): The name of the recognition model.

    Returns:
        tuple: A tuple containing the loaded detection models (`det_model` and `weight_det_model`) and the loaded recognition model `_rec_model`.

    Raises:
        None

    Examples:
        >>> preprocess_config("yolov8-sl", "w-yolov8-sl", "svtr_p")
        (<YOLO object>, <YOLO object>, <REC_MODEL object>)
    """
    det_model = YOLO_SL(f'weights/{det_model_1}.pt')
    print(f"✅Successfually loaded: {det_model_1} model!!!")
    weight_det_model = YOLO_SL(f'weights/{det_model_2}.pt')
    print(f"✅Successfually loaded {det_model_2} model!!!")
    _rec_model = SVTR_G('rec/cfg/svtr_g.yml', f'weights/{rec_model}')
    print(f"✅Successfually loaded {rec_model} model!!!\n")

    return det_model, weight_det_model, _rec_model


def _get_cls_boxes(results):
        """
        Get the class boxes from the given results.

        Parameters:
            results (object): The results object containing the boxes and names.

        Returns:
            list: A list of class boxes, where each box is represented as a list containing the name, confidence, 
                  coordinates, and class index.
        """
        names = results.names
        res = []
        for d in reversed(results.boxes):
            c, conf, id = (int(d.cls), float(d.conf), None if d.id is None else int(d.id.item()),)
            name = ("" if id is None else f"id:{id} ") + names[c]
            res.append([name, f"{conf:.2f}", d.xyxy.squeeze().tolist(), c])

        return res



if __name__ == '__main__':

    device = "gpu" if torch.cuda.is_available() else "cpu"

    img_path = "select_img/7058.jpg"   # Image path: can be a single image or a directory of images
    save_dir = "results"    # Directory to save the results
    os.makedirs(save_dir, exist_ok=True)
    
    det_model, weight_det_model, rec_model = _preprocess_config('YOLOv8-SL', 'SL-TD', 'SVTR-G')
    
    main()







