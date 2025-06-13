import torch
from torch import nn
import numpy as np
from mq_glip_demo import MQGLIPDemo
from typing import List, Union, Optional
import PIL

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../sam2')))
# from ..sam2.sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2_video_predictor
from onevision.utils.rle import rle_encode, robust_rle_encode
from onevision.models.detr.video_tracking_with_prompt_utils import mask_to_box


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
    
    def forward_single_prompt(
        self, 
        text_prompt: str, 
        box_prompt: List[List[int]], 
        video_dir: str, 
        frame_idx_to_run_image_grounding=0,
        box_prompt_mode="xyxy"
    ):
        """
        text_prompt: str, the text prompt for MQ-GLIP
        box_prompt: List[Box], each Box is a list of 4 floats denoting a bounding box with 4 coordinates, 
            formatted indicated by `box_prompt_mode`
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

    def forward(
            self, 
            text_prompt_list: List[str],
            box_prompt_list: List[List[List[float]]], 
            video_dir_or_loaded_frames: Union[str, List[PIL.Image.Image]], 
            frame_idx_to_run_image_grounding: int = 0,
            box_prompt_mode: str = "xyxy",
            text_prompt_ids: Optional[List[int]] = None
        ):
        """
        Run MQ-GLIP and SAM2 for multiple text and box prompts pairs.
        
        text_prompt_list: List[str], list of text prompts for MQ-GLIP
        box_prompt_list: List[List[Box]], each Box is a list of 4 floats denoting a bounding box with 4 coordinates, 
            formatted indicated by `box_prompt_mode`
        video_dir: str, path to the video
        box_prompt_mode: str, the format of the prompt boxes, either "xywh" or "xyxy"
        """
        assert len(text_prompt_list) == len(box_prompt_list), "text_prompt_list and box_prompt_list must have the same length"
        # setup data
        if isinstance(video_dir_or_loaded_frames, str):
            video_frame_paths = self.get_frame_paths(video_dir_or_loaded_frames)
            # Load the image
            video_frames = [self.mq_glip_model.load_image(p) for p in video_frame_paths]
        else:
            video_frames = video_dir_or_loaded_frames
        
        # Run image grounding for each pair of text-box prompt
        prompt_id2obj_ids = {} # keep note of which boxes belong to which prompts
        grounding_results = {"bbox": [], "scores": [], "labels": []}
        image_for_grounding = video_frames[frame_idx_to_run_image_grounding]
        prompt_ids = text_prompt_ids if text_prompt_ids is not None else list(range(len(text_prompt_list)))
        for prompt_id, text_prompt, box_prompt in zip(prompt_ids, text_prompt_list, box_prompt_list):
            _grounding_results = self.run_image_grounding(
                image_for_grounding, 
                text_prompt, 
                box_prompt=box_prompt, 
                box_prompt_mode=box_prompt_mode
            )
            st, cur_len = len(grounding_results["bbox"]), len(_grounding_results["bbox"])
            prompt_id2obj_ids[prompt_id] = list(range(st, st + cur_len))
            grounding_results["labels"].extend([prompt_id] * cur_len)
            grounding_results["bbox"].extend(_grounding_results["bbox"])
            grounding_results["scores"].extend(_grounding_results["scores"])

        # Run SAM2 with the detected boxes
        tracking_results = self.run_sam2_tracking(
            video_frames, 
            grounding_results, 
            grounding_frame_idx=frame_idx_to_run_image_grounding
        )
        return {
            "grounding_results": grounding_results,
            "tracking_results": tracking_results,
            "video_frames": video_frames,
            "prompt_id2obj_ids": prompt_id2obj_ids
        }
        
    def prepare_results_in_sam3_eval_format(self, results):
        preds = {} # ['scores', 'labels', 'boxes', 'masks_rle', 'per_frame_scores']
        preds["scores"] = torch.tensor(results["grounding_results"]["scores"], device=self.device) # (n_tracklet,)
        preds["labels"] = torch.tensor(results["grounding_results"]["labels"], device=self.device) # (n_tracklet,)
        # masks
        # boxes
        return preds

    def run_sam2_tracking(self, video_path_or_load_frames, grounding_results, grounding_frame_idx, mask_to_numpy=True):
        """
        Run SAM2 tracking on the video frames using the detected boxes from MQ-GLIP.
        
        video_path_or_load_frames: str path to the video or List[PIL.Image.Image] in RGB mode, 
        grounding_results: dict, results from MQ-GLIP containing 'bbox' and 'scores'
        grounding_frame_idx: int, the index of the frame where the grounding results were obtained
        
        Returns:
            sam2_results: List[dict], each dict contains the results for each frame
        """
        num_boxes = len(grounding_results['bbox'])
        if num_boxes == 0:
            return {}

        # has boxes
        inference_state = self.sam2_predictor.init_state(
            video_path_or_load_frames=video_path_or_load_frames
        )
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
        # the propagate_in_video will not process the frames before grounding_frame_idx
        # so we need a reverse loop to fill in the missing frames
        if grounding_frame_idx != 0:
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_predictor.propagate_in_video(
                    inference_state,
                    reverse=True
                ):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() if mask_to_numpy else out_mask_logits[i]
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

        return video_segments
        

    def run_image_grounding(
            self, 
            original_image: PIL.Image.Image, 
            text_prompt: str, 
            box_prompt: List[List[float]]=None, 
            box_prompt_mode: str ="xyxy"
        ):
        """
        Run image grounding on a single image.
        
        original_image: original Image
        text_prompt: str, the text prompt for MQ-GLIP
        box_prompt: List[Box], each Box is a list of 4 floats denoting a bounding box with 4 coordinates, 
            formatted indicated by `box_prompt_mode`
        box_prompt_mode: str, the format of the prompt boxes, either "xywh" or "xyxy"
        
        Returns:
            results: a dict with two keys,
                - bbox: List[List[int]], each inner list is a bounding box with 4 coordinates,
                formatted as xyxy, indicating the detected boxes in the image. May be empty if no
                boxes are detected.
                - scores: List[float], the scores of the detected boxes.
        """        
        # Prepare the image
        image_list = self.mq_glip_model.prepare_image(np.array(original_image))
        
        # Prepare the caption
        _, positive_map_label_to_token = self.mq_glip_model.prepare_caption([text_prompt, ])
        
        # Run MQ-GLIP model
        vision_enabled = True if box_prompt is not None else False
        preds = self.mq_glip_model(
            images=image_list,
            captions=[text_prompt, ],
            positive_map=positive_map_label_to_token,
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

