# Copyright (c) Lin Song, All Rights Reserved.
from typing import List

import torch
import numpy as np
import supervision as sv
from inference.models import YOLOWorld as YOLOWorldImpl


class YOLOWorld:
    """ YOLOWorld node class """

    RETURN_TYPES = ('DETECTIONS',)
    RETURN_NAMES = ('detections',)
    FUNCTION = 'yoloworld_image'
    CATEGORY = 'YOLOWorld'

    @classmethod
    def INPUT_TYPES(cls):
        return dict(
            required=dict(
                model=('YOLOWORLDMODEL',),
                images=('IMAGE',),
                confidence_threshold=('FLOAT', dict(
                    default=0.05, min=0, max=1, step=0.01)),
                nms_iou_threshold=('FLOAT', dict(
                    default=0.3, min=0, max=1, step=0.01)),
                with_class_agnostic_nms=('BOOLEAN', dict(default=False)),
            )
        )

    def yoloworld_image(
            self,
            images: List[torch.Tensor],
            model: torch.nn.Module,
            confidence_threshold: float,
            nms_iou_threshold: float,
            with_class_agnostic_nms: bool) -> List[sv.Detections]:
        output = []
        for img in images:
            img = (255 * img.cpu().numpy()).astype(np.uint8)
            results = model.infer(
                img, confidence=confidence_threshold)
            detections = sv.Detections.from_inference(results)
            detections = detections.with_nms(
                class_agnostic=with_class_agnostic_nms,
                threshold=nms_iou_threshold
            )
            output.append(detections)
        return [output]


class YOLOWorld_ModelLoader:
    """ YOLOWorld Model Loader node class """

    RETURN_TYPES = ('YOLOWORLDMODEL',)
    RETURN_NAMES = ('model',)
    FUNCTION = 'load_yolo_world_model'
    CATEGORY = 'YOLOWorld'

    @classmethod
    def INPUT_TYPES(cls):
        default_categories = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        return dict(
            required=dict(
                model_id=(
                     ['yolo_world/v2-x', 'yolo_world/v2-l', 'yolo_world/v2-m',
                      'yolo_world/v2-s', 'yolo_world/l', 'yolo_world/m',
                      'yolo_world/s'],),
                categories=(
                    'STRING', dict(
                        display='Categories',
                        default=','.join(default_categories),
                        multiline=False))
            )
        )

    def process_categories(self, categories: str) -> List[str]:
        return [category.strip().lower() for category in categories.split(',')]

    def load_yolo_world_model(
            self,
            model_id: str,
            categories: str) -> List[torch.nn.Module]:
        model = YOLOWorldImpl(model_id=model_id)
        categories = self.process_categories(categories)
        model.set_classes(categories)
        return [model]


class YOLOWorld_Display:
    """ YOLOWorld Model Loader node class """

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('images',)
    FUNCTION = 'annotate_image'
    CATEGORY = 'YOLOWorld'

    @classmethod
    def INPUT_TYPES(cls):
        return dict(
            required=dict(
                images=('IMAGE',),
                detections=('DETECTIONS',),
                with_confidence=('BOOLEAN', dict(default=True)),
                thickness=('INT', dict(
                    default=2, min=1, max=10, step=1)),
                text_thickness=('INT', dict(
                    default=2, min=1, max=10, step=1)),
                text_scale=('FLOAT', dict(
                    default=1.0, min=0.1, max=2, step=0.1)),
            )
        )

    def annotate_image(
        self,
        images: List[np.ndarray],
        detections: sv.Detections,
        with_confidence: bool,
        thickness: int,
        text_thickness: int,
        text_scale: float
    ) -> np.ndarray:
        output_images = []
        for img, det in zip(images, detections):
            labels = []
            img = (255 * img.cpu().numpy()).astype(np.uint8)
            for name, conf in zip(det.data['class_name'],
                                  det.confidence):
                labels.append(
                    f'{name}: {conf:.3f}' if with_confidence else name)

            bounding_box_annotator = sv.BoundingBoxAnnotator(
                thickness=thickness)
            label_annotator = sv.LabelAnnotator(
                text_thickness=text_thickness, text_scale=text_scale)
            output_image = bounding_box_annotator.annotate(img, det)
            output_image = label_annotator.annotate(
                output_image, det, labels=labels)
            output_image = torch.from_numpy(output_image.astype(
                np.float32) / 255.0).unsqueeze(0)
            output_images.append(output_image)
        output_images = torch.cat(output_images, dim=0)
        return [output_images]


NODE_CLASS_MAPPINGS = dict(
    YOLOWorld=YOLOWorld,
    YOLOWorld_ModelLoader=YOLOWorld_ModelLoader,
    YOLOWorld_Display=YOLOWorld_Display
)


NODE_DISPLAY_NAME_MAPPINGS = dict(
    YOLOWorld='YOLO-World',
    YOLOWorld_ModelLoader='YOLO-World Model Loader',
    YOLOWorld_Display='YOLO-World Display'
)
