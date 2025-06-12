import torch
from torch import nn
import numpy as np
from mq_glip_demo import MQGLIPDemo
from typing import List

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../sam2')))
# from ..sam2.sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2_video_predictor


class MQGLIPSam2JointModel(nn.Module):
    def __init__(
        self, 
        mq_glip_base_dir="/home/jielei/MQ-Det",
        sam2_checkpoint="/home/jielei/sam2/checkpoints/sam2.1_hiera_large.pt",
        sam2_model_config="configs/sam2.1/sam2.1_hiera_l.yaml", # uses relative path within sam2 to avoid hydra error
        local_rank=0,
        mq_glip_threshold=0.5
    ):
        """Assume we are on a single GPU"""
        super(MQGLIPSam2JointModel, self).__init__()
        self.device = torch.device(f"cuda:{local_rank}")
        self.mq_glip_threshold = mq_glip_threshold
        # MQ-GLIP
        mq_glip_cfg = self.get_mq_glip_cfg(mq_glip_base_dir, local_rank=local_rank)
        self.mq_glip_model = MQGLIPDemo(mq_glip_cfg, device=self.device)
        # SAM2
        self.sam2_predictor = build_sam2_video_predictor(
            sam2_model_config, sam2_checkpoint, device=self.device
        )
    
    def forward(
        self, 
        text_prompt: str, 
        box_prompt: List[List[int]], 
        video_dir: str, 
        frame_idx_to_run_image_grounding=0,
        box_prompt_mode="xyxy"
    ):
        """
        text_prompt: str, the text prompt for MQ-GLIP
        box_prompt: List[List[int]], each inner list is a bounding box with 4 coordinates, 
            formatted as xyxy, indicated by `box_prompt_mode`
        video_dir: str, path to the video
        frame_idx_to_run_image_grounding: int, the index of the frame to run image grounding
        box_prompt_mode: str, the format of the prompt boxes, either "xywh" or "xyxy"
        """
        video_frame_paths = self.get_frame_paths(video_dir)
        image_path_for_grounding = video_frame_paths[frame_idx_to_run_image_grounding]
        # Run image grounding
        grounding_results = self.run_image_grounding(
            image_path_for_grounding, 
            text_prompt, 
            box_prompt=box_prompt, 
            box_prompt_mode=box_prompt_mode
        )
        # Run SAM2 with the detected boxes
        tracking_results = self.run_sam2_tracking(
            video_dir, 
            grounding_results, 
            grounding_frame_idx=frame_idx_to_run_image_grounding
        )
        return {
            "grounding_results": grounding_results,
            "tracking_results": tracking_results,
            "video_frame_paths": video_frame_paths
        }

    def run_sam2_tracking(self, video_dir, grounding_results, grounding_frame_idx):
        """
        Run SAM2 tracking on the video frames using the detected boxes from MQ-GLIP.
        
        video_dir: str, path to the video
        grounding_results: dict, results from MQ-GLIP containing 'bbox' and 'scores'
        grounding_frame_idx: int, the index of the frame where the grounding results were obtained
        
        Returns:
            sam2_results: List[dict], each dict contains the results for each frame
        """
        num_boxes = len(grounding_results['bbox'])
        if num_boxes == 0:
            return {}

        # has boxes
        inference_state = self.sam2_predictor.init_state(video_path=video_dir)
        # segment object with box prompt
        for obj_id, box in enumerate(grounding_results['bbox']):
            box = np.array(box, dtype=np.float32)
            _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=grounding_frame_idx,
                obj_id=obj_id,
                box=box,
            )
        
        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        # TODO: add image grounding scores to the video_segments as the video-level score?
        return video_segments
        
        
    def run_image_grounding(self, image_path, text_prompt, box_prompt=None, box_prompt_mode="xyxy"):
        """
        Run image grounding on a single image.
        
        image_path: str, path to the image
        text_prompt: str, the text prompt for MQ-GLIP
        box_prompt: List[List[int]], each inner list is a bounding box with 4 coordinates,
            formatted as xyxy, indicated by `box_prompt_mode`        
        box_prompt_mode: str, the format of the prompt boxes, either "xywh" or "xyxy"
        
        Returns:
            results: a dict with two keys,
                - bbox: List[List[int]], each inner list is a bounding box with 4 coordinates,
                formatted as xyxy, indicating the detected boxes in the image. May be empty if no
                boxes are detected.
                - scores: List[float], the scores of the detected boxes.
        """
        # Load the image
        original_image = self.mq_glip_model.load_image(image_path)
        
        # Prepare the image
        image_list = self.mq_glip_model.prepare_image(original_image)
        
        # Prepare the caption
        positive_map, _ = self.mq_glip_model.prepare_caption(text_prompt)
        
        # Run MQ-GLIP model
        vision_enabled = True if box_prompt is not None else False
        preds = self.mq_glip_model(
            images=image_list,
            captions=[text_prompt],
            positive_map=positive_map,
            prompt_boxes=box_prompt,
            vision_enabled=vision_enabled,
            prompt_box_mode=box_prompt_mode
        )
        
        preds_processed = self.mq_glip_model._post_process(preds, threshold=self.mq_glip_threshold)
        preds_processed_resize = preds_processed.resize(original_image.size) # width, height
        detected_boxes = preds_processed_resize.bbox.cpu().tolist()
        results = {
            "bbox": detected_boxes,
            "scores": preds_processed_resize.get_field("scores").cpu().tolist(),
        }
        return results

    def get_frame_paths(self, video_dir: str):
        # scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        video_frame_paths = [os.path.join(video_dir, p) for p in frame_names]
        if len(video_frame_paths) == 0:
            raise ValueError(f"No JPEG frames found in {video_dir}.")
        return video_frame_paths    
        
    def get_mq_glip_cfg(self, mq_glip_base_dir, local_rank=0):
        """MQ-GLIP-L config
        TODO: how does the local_rank and num_gpus affect the model?
        """
        from maskrcnn_benchmark.config import cfg
        cfg.local_rank = local_rank
        cfg.num_gpus = 1
        config_file = os.path.join(mq_glip_base_dir, "configs/pretrain/mq-glip-l.yaml")
        cfg.merge_from_file(config_file)
        additional_model_config = os.path.join(mq_glip_base_dir, "configs/vision_query_5shot/lvis_minival_L.yaml")
        cfg.merge_from_file(additional_model_config)
        
        opts = [
            "MODEL.WEIGHT", os.path.join(mq_glip_base_dir, "MODEL/mq-glip-l"),
            "TEST.IMS_PER_BATCH", "1",
            "VISION_QUERY.QUERY_BANK_PATH", os.path.join(mq_glip_base_dir, "MODEL/lvis_query_5_pool7_sel_large.pth"),
        ]
        cfg.merge_from_list(opts)
        
        cfg.freeze()
        return cfg        

