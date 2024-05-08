import copy
import importlib

from paddle.jit import to_static
from paddle.static import InputSpec

from paddle import nn
from rec.modeling.transforms import build_transform
from rec.modeling.backbones import build_backbone
from rec.modeling.necks import build_neck
from rec.modeling.heads import build_head


def build_model(config):
    config = copy.deepcopy(config)
    if "name" not in config:
        arch = BaseModel(config)
    else:
        name = config.pop("name")
        mod = importlib.import_module(__name__)
        arch = getattr(mod, name)(config)
    return arch


def apply_to_static(model, config, logger):
    if config["Global"].get("to_static", False) is not True:
        return model
    assert "d2s_train_image_shape" in config[
        "Global"], "d2s_train_image_shape must be assigned for static training mode..."
    supported_list = [
        "SVTR"
    ]
    if config["Architecture"]["algorithm"] in ["Distillation"]:
        algo = list(config["Architecture"]["Models"].values())[0]["algorithm"]
    else:
        algo = config["Architecture"]["algorithm"]
    assert algo in supported_list, f"algorithms that supports static training must in in {supported_list} but got {algo}"

    specs = [
        InputSpec(
            [None] + config["Global"]["d2s_train_image_shape"], dtype='float32')
    ]

    if algo == "SVTR":
        specs.append([
            InputSpec(
                [None, config["Global"]["max_text_length"]], dtype='int64'),
            InputSpec(
                [None], dtype='int64')
        ])
    model = to_static(model, input_spec=specs)
    logger.info("Successfully to apply @to_static with specs: {}".format(specs))
    return model


class BaseModel(nn.Layer):
    def __init__(self, config):
        """
        the module for OCR.
        args:
            config (dict): the super parameters for module.
        """
        super(BaseModel, self).__init__()
        in_channels = config.get('in_channels', 3)
        model_type = config['model_type']
        # build transfrom,
        # for rec, transfrom can be TPS,None
        # for det and cls, transfrom shoule to be None,
        # if you make model differently, you can use transfrom in det and cls
        if 'Transform' not in config or config['Transform'] is None:
            self.use_transform = False
        else:
            self.use_transform = True
            config['Transform']['in_channels'] = in_channels
            self.transform = build_transform(config['Transform'])
            in_channels = self.transform.out_channels

        # build backbone, backbone is need for del, rec and cls
        if 'Backbone' not in config or config['Backbone'] is None:
            self.use_backbone = False
        else:
            self.use_backbone = True
            config["Backbone"]['in_channels'] = in_channels
            self.backbone = build_backbone(config["Backbone"], model_type)
            in_channels = self.backbone.out_channels

        # build neck
        # for rec, neck can be cnn,rnn or reshape(None)
        # for det, neck can be FPN, BIFPN and so on.
        # for cls, neck should be none
        if 'Neck' not in config or config['Neck'] is None:
            self.use_neck = False
        else:
            self.use_neck = True
            config['Neck']['in_channels'] = in_channels
            self.neck = build_neck(config['Neck'])
            in_channels = self.neck.out_channels
            
        # # build head, head is need for det, rec and cls
        if 'Head' not in config or config['Head'] is None:
            self.use_head = False
        else:
            self.use_head = True
            config["Head"]['in_channels'] = in_channels
            self.head = build_head(config["Head"])

        self.return_all_feats = config.get("return_all_feats", False)

    def forward(self, x, data=None):
        y = dict()
        if self.use_transform:
            x = self.transform(x)
        if self.use_backbone:
            x = self.backbone(x)
        if isinstance(x, dict):
            y.update(x)
        else:
            y["backbone_out"] = x
        final_name = "backbone_out"
        if self.use_neck:
            x = self.neck(x)
            if isinstance(x, dict):
                y.update(x)
            else:
                y["neck_out"] = x
            final_name = "neck_out"
        if self.use_head:
            x = self.head(x, targets=data)
            # for multi head, save ctc neck out for udml
            if isinstance(x, dict) and 'ctc_neck' in x.keys():
                y["neck_out"] = x["ctc_neck"]
                y["head_out"] = x
            elif isinstance(x, dict):
                y.update(x)
            else:
                y["head_out"] = x
            final_name = "head_out"
        if self.return_all_feats:
            if self.training:
                return y
            elif isinstance(x, dict):
                return x
            else:
                return {final_name: x}
        else:
            return x