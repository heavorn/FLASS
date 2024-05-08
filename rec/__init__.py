import os
os.environ['GLOG_minloglevel'] = '10000'

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import paddle

from rec.data import create_operators, transform
from rec.modeling.architectures import build_model
from rec.modeling.postprocess import build_post_process
from rec.utils.save_load import load_pretrained_params
from rec.utils.logging import get_logger
import rec.engine.engine as program
import numpy as np

class REC_MODEL:
    def __init__(self, cfg, pretrain_model=None):
        super(REC_MODEL, self).__init__()
        self.config = program.load_config(cfg)
        self.global_config = self.config['Global']
        self.pretrain_model = pretrain_model
        self.logger = get_logger()
        self.post_process_class = build_post_process({'name': 'CTCLabelDecode'})
        self.model = self.build_model()  # Store the model as an instance variable

    def build_model(self):
        # pdb.set_trace()
        char_num = len(getattr(self.post_process_class, 'character'))
        self.config["Architecture"]["Head"]["out_channels"] = char_num
        model = build_model(self.config['Architecture'])

        if self.pretrain_model is not None:
            load_pretrained_params(model, self.pretrain_model)
        
        # Return an instance of REC_MODEL with the model loaded
        return model

    def predict(self, crop_img):
        # create data ops
        transforms = [
            # {'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}},
            {'SVTRRecResizeImg': {'image_shape': [3, 64, 256], 'padding': False}},
            {'KeepKeys': {'keep_keys': ['image']}}]
        
        ops = create_operators(transforms)
        self.model.eval()
        data = {'image': crop_img}
        batch = transform(data, ops)
        images = np.expand_dims(batch[0], axis=0)
        images = paddle.to_tensor(images)
        preds = self.model(images)
        post_result = self.post_process_class(preds)
        return (post_result[0][0], post_result[0][1])
    
    def val(self, crop_img, device):
        device = paddle.set_device(device)
        # create data ops
        transforms = [
            # {'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}},
            {'SVTRRecResizeImg': {'image_shape': [3, 64, 256], 'padding': False}},
            {'KeepKeys': {'keep_keys': ['image']}}]
        
        ops = create_operators(transforms)
        self.model.eval()
        # batches = []
        batches = [transform({'image': img}, ops) for img in crop_img]
        # for img in crop_img:
        #     # import pdb; pdb.set_trace()
        #     print(f"{len(crop_img)} has {len(img)} words")
        #     data = {'image': img}
        #     batches.append(transform(data, ops))
        images = np.stack([batch[0] for batch in batches], axis=0)
        
        images = paddle.to_tensor(images)
        # import pdb; pdb.set_trace()
        preds = self.model(images)
        post_results = self.post_process_class(preds)

        return post_results