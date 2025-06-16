import torch
from torch import nn
import nltk
import inflect
from transformers import AutoTokenizer
import requests
from io import BytesIO
from PIL import Image
from torchvision import transforms as T
import pdb
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList

engine = inflect.engine()
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import os


def create_positive_map_label_to_token_from_positive_map(positive_map, plus=0):
    positive_map_label_to_token = {}
    for i in range(len(positive_map)):
        positive_map_label_to_token[i + plus] = torch.nonzero(positive_map[i], as_tuple=True)[0].tolist()
    return positive_map_label_to_token


def create_positive_map(tokenized, tokens_positive):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            try:
                beg_pos = tokenized.char_to_token(beg)
                end_pos = tokenized.char_to_token(end - 1)
            except Exception as e:
                print("beg:", beg, "end:", end)
                print("token_positive:", tokens_positive)
                raise e
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos: end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


class MQGLIPDemo(nn.Module):
    def __init__(self,
                 cfg,
                 min_image_size=800,
                 model=None,
                 device="cuda"
                 ):
        super(MQGLIPDemo, self).__init__()
        self.cfg = cfg.clone()
        self.color = 255 
        self.min_image_size = min_image_size
        self.cpu_device = torch.device("cpu")
        # if device is None:
        #     device = torch.device(cfg.MODEL.DEVICE)
        # self.device = device
        
        self.model = self.prepare_model(model, cfg, device=device)

        self.tokenizer = self.build_tokenizer()
        self.transforms = self.build_transform()

    @property
    def device(self):
        return next(self.parameters()).device

    def prepare_model(self, model, cfg, device="cuda", mode="eval"):
        if model is None:
            model = build_detection_model(cfg)
            checkpointer = DetectronCheckpointer(
                cfg, model, save_dir=cfg.OUTPUT_DIR)
            _ = checkpointer.load(cfg.MODEL.WEIGHT)
        if mode == "eval":
            model.eval()
        model = model.to(device)
        return model          

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size) if self.min_image_size is not None else lambda x: x,
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def build_tokenizer(self):
        cfg = self.cfg
        tokenizer = None
        if os.path.basename(cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE) == "bert-base-uncased":
            tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE)
        elif cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
            from transformers import CLIPTokenizerFast
            if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
                tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                              from_slow=True, mask_token='ðŁĴĳ</w>')
            else:
                tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                              from_slow=True)
        return tokenizer
    

    def prepare_caption(self, original_caption):
        # caption
        if isinstance(original_caption, list):
            # we directly provided a list of category names
            caption_string = ""
            tokens_positive = []
            seperation_tokens = " . "
            for word in original_caption:
                
                tokens_positive.append([len(caption_string), len(caption_string) + len(word)])
                caption_string += word
                caption_string += seperation_tokens
            
            tokenized = self.tokenizer([caption_string], return_tensors="pt")
            tokens_positive = [tokens_positive]

            original_caption = caption_string
        else:
            raise NotImplementedError

        # process positive map
        positive_map = create_positive_map(tokenized, tokens_positive)

        if self.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
            plus = 1
        else:
            plus = 0

        positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map, plus=plus)
        # self.plus = plus
        # self.positive_map_label_to_token = positive_map_label_to_token

        return positive_map, positive_map_label_to_token

    def load_image(self, url_or_image_path, convert_bgr=True):
        if url_or_image_path.startswith("http://") or url_or_image_path.startswith("https://"):
            response = requests.get(url_or_image_path)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(url_or_image_path).convert("RGB")
        # # convert to BGR format
        # if convert_bgr:
        #     image = np.array(image)[:, :, [2, 1, 0]]
        return image

    def prepare_image(self, original_image):
        image = self.transforms(original_image)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        return image_list

    def forward(
        self, 
        images, 
        targets=None, 
        captions=None, 
        positive_map=None,
        prompt_boxes=None,
        vision_enabled=False,
        prompt_box_mode="xyxy" 
        ):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            prompt_boxes: List[List[int]], each inner list is a bounding box with 4 coordinates, formatted as xyxy
            captions (list[str]): list of captions for each image, used for language embedding
            prompt_box_mode (str): the format of the prompt boxes, either "xywh" or "xyxy"

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        images = to_image_list(images)
        # batch_size = images.tensors.shape[0]
        device = images.tensors.device

        # visual embedding
        swint_feature_c4 = None
        visual_features = self.model.backbone(images.tensors)

        # query embedding
        # if self.cfg.VISION_QUERY.ENABLED:
        if vision_enabled:
            assert images.tensors.shape[0]==1 # TODO: Only query batch size = 1 for test
            vision_inputs_in_language_backbone = self.get_vision_inputs_for_language_backbone(
                visual_features, positive_map, prompt_boxes, image_size=images.tensors.shape[-2:], box_mode=prompt_box_mode
            )
        else:
            vision_inputs_in_language_backbone={
                'vision': None, 
                'images': None, 
                'vision_attention_mask': None, 
                'batched_pos_category_map': None
            }

        # language embedding
        language_dict_features = {}
        if captions is not None:
            tokenized = self.model.tokenizer.batch_encode_plus(captions,
                                                        max_length=self.model.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                                        padding='max_length' if self.model.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                        return_special_tokens_mask=True,
                                                        return_tensors='pt',
                                                        truncation=True).to(device)

            input_ids = tokenized.input_ids
            
            tokenizer_input = {"input_ids": input_ids,
                            "attention_mask": tokenized.attention_mask,
                            "vision_inputs": vision_inputs_in_language_backbone}

            language_dict_features = self.model.language_backbone(tokenizer_input)
            
            language_dict_features["mlm_labels"] = None

        # rpn force boxes
        if targets:
            targets = [target.to(device)
                       for target in targets if target is not None]

        proposals, proposal_losses, fused_visual_features = self.model.rpn(
            images, visual_features, targets, language_dict_features, positive_map,
            captions, swint_feature_c4
        )

        # RPN-only models don't have roi_heads
        x = visual_features
        result = proposals
        detector_losses = {}

        # remove batch dim
        result = result[0] 
        result = result.to(self.cpu_device)
        return result        

    def create_attention_map(self, num_boxes, all_map):
        """
        num_boxes: int
        all_map: torch.tensor (1, 256) indicating the attention weights of the current label across text length 256
        query_attetion_masks: torch.tensor (1, num_boxes, 256) indicating the attention mask for each box
        """
        query_attetion_masks = all_map.expand(num_boxes, -1)
        query_attetion_masks[query_attetion_masks != 0] = 1
        # import ipdb; ipdb.set_trace()
        return query_attetion_masks[None, ...] # # 1, num_boxes, 256

    def extract_visual_prompt_feature(self, visual_features, box_list, image_size, box_mode="xywh"):
        """
        box_list: List[List[int]], make sure `box_list` is defined for `image_size`
        """
        # prepare box
        assert box_mode in ["xywh", "xyxy"], f"box_mode should be one of ['xywh', 'xyxy'], but got {box_mode}"
        box_list = torch.as_tensor(box_list).reshape(-1, 4).to(self.device)  # guard against no boxes
        if box_mode == "xywh":
            box_list = [BoxList(box_list, image_size, mode="xywh").convert("xyxy"), ]
        else: # xyxy
            box_list = [BoxList(box_list, image_size, mode="xyxy"), ]
        # run ROI pooling
        query_feats=self.model.pooler(visual_features, box_list) # num_boxes, num_channels, pooler_size, pooler_size
        query_feats=query_feats[None, ...] # 1, num_boxes, num_channels, pooler_size, pooler_size
        query_feats = query_feats.mean(dim=[-2,-1]) # batch_size, num_boxes, num_channels
        return query_feats

    def get_vision_inputs_for_language_backbone(self, visual_features, positive_map, boxes, image_size, box_mode="xyxy"):
        labels_in_caption, all_map = self.model.get_labels_and_maps_from_positive_map(
            positive_map, dtype=visual_features[0].dtype
        )
        query_features = self.extract_visual_prompt_feature(visual_features, boxes, image_size, box_mode="xyxy")
        query_attetion_masks = self.create_attention_map(len(boxes), all_map) 
        
        vision_inputs_in_language_backbone={
            'vision': query_features,
            'images': self.model.flatten_fpn_features(visual_features), 
            'vision_attention_mask': query_attetion_masks, 
            'batched_pos_category_map': None
            }
        return vision_inputs_in_language_backbone

    def _post_process(self, predictions, threshold=0.5):
        # copied GLIP_DEMO
        scores = predictions.get_field("scores")
        labels = predictions.get_field("labels").tolist()
        thresh = scores.clone()
        for i, lb in enumerate(labels):
            thresh[i] = threshold
        keep = torch.nonzero(scores > thresh).squeeze(1)
        predictions = predictions[keep]

        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]
