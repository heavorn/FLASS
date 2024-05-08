import math
import paddle
from paddle import nn, ParamAttr
from paddle.nn import functional as F
import numpy as np
import itertools


def build_transform(config):

    support_dict = ['STN_ON']

    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'transform only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class


def conv3x3_block(in_channels, out_channels, stride=1):
    n = 3 * 3 * out_channels
    w = math.sqrt(2. / n)
    conv_layer = nn.Conv2D(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        weight_attr=nn.initializer.Normal(
            mean=0.0, std=w),
        bias_attr=nn.initializer.Constant(0))
    block = nn.Sequential(conv_layer, nn.BatchNorm2D(out_channels), nn.ReLU())
    return block


class STN(nn.Layer):
    def __init__(self, in_channels, num_ctrlpoints, activation='none'):
        super(STN, self).__init__()
        self.in_channels = in_channels
        self.num_ctrlpoints = num_ctrlpoints
        self.activation = activation
        self.stn_convnet = nn.Sequential(
            conv3x3_block(in_channels, 32),  #32x64
            nn.MaxPool2D(
                kernel_size=2, stride=2),
            conv3x3_block(32, 64),  #16x32
            nn.MaxPool2D(
                kernel_size=2, stride=2),
            conv3x3_block(64, 128),  # 8*16
            nn.MaxPool2D(
                kernel_size=2, stride=2),
            conv3x3_block(128, 256),  # 4*8
            nn.MaxPool2D(
                kernel_size=2, stride=2),
            conv3x3_block(256, 256),  # 2*4,
            nn.MaxPool2D(
                kernel_size=2, stride=2),
            conv3x3_block(256, 256))  # 1*2
        self.stn_fc1 = nn.Sequential(
            nn.Linear(
                2 * 256,
                512,
                weight_attr=nn.initializer.Normal(0, 0.001),
                bias_attr=nn.initializer.Constant(0)),
            nn.BatchNorm1D(512),
            nn.ReLU())
        fc2_bias = self.init_stn()
        self.stn_fc2 = nn.Linear(
            512,
            num_ctrlpoints * 2,
            weight_attr=nn.initializer.Constant(0.0),
            bias_attr=nn.initializer.Assign(fc2_bias))

    def init_stn(self):
        margin = 0.01
        sampling_num_per_side = int(self.num_ctrlpoints / 2)
        ctrl_pts_x = np.linspace(margin, 1. - margin, sampling_num_per_side)
        ctrl_pts_y_top = np.ones(sampling_num_per_side) * margin
        ctrl_pts_y_bottom = np.ones(sampling_num_per_side) * (1 - margin)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        ctrl_points = np.concatenate(
            [ctrl_pts_top, ctrl_pts_bottom], axis=0).astype(np.float32)
        if self.activation == 'none':
            pass
        elif self.activation == 'sigmoid':
            ctrl_points = -np.log(1. / ctrl_points - 1.)
        ctrl_points = paddle.to_tensor(ctrl_points)
        fc2_bias = paddle.reshape(
            ctrl_points, shape=[ctrl_points.shape[0] * ctrl_points.shape[1]])
        return fc2_bias

    def forward(self, x):
        x = self.stn_convnet(x)
        batch_size, _, h, w = x.shape
        x = paddle.reshape(x, shape=(batch_size, -1))
        img_feat = self.stn_fc1(x)
        x = self.stn_fc2(0.1 * img_feat)
        if self.activation == 'sigmoid':
            x = F.sigmoid(x)
        x = paddle.reshape(x, shape=[-1, self.num_ctrlpoints, 2])
        return img_feat, x


class STN_ON(nn.Layer):
    def __init__(self, in_channels, tps_inputsize, tps_outputsize,
                 num_control_points, tps_margins, stn_activation):
        super(STN_ON, self).__init__()
        self.tps = TPSSpatialTransformer(
            output_image_size=tuple(tps_outputsize),
            num_control_points=num_control_points,
            margins=tuple(tps_margins))
        self.stn_head = STN(in_channels=in_channels,
                            num_ctrlpoints=num_control_points,
                            activation=stn_activation)
        self.tps_inputsize = tps_inputsize
        self.out_channels = in_channels

    def forward(self, image):
        stn_input = paddle.nn.functional.interpolate(
            image, self.tps_inputsize, mode="bilinear", align_corners=True)
        stn_img_feat, ctrl_points = self.stn_head(stn_input)
        x, _ = self.tps(image, ctrl_points)
        return x
    

def grid_sample(input, grid, canvas=None):
    input.stop_gradient = False

    is_fp16 = False
    if grid.dtype != paddle.float32:
        data_type = grid.dtype
        input = input.cast(paddle.float32)
        grid = grid.cast(paddle.float32)
        is_fp16 = True
    output = F.grid_sample(input, grid)
    if is_fp16:
        output = output.cast(data_type)
        grid = grid.cast(data_type)

    if canvas is None:
        return output
    else:
        input_mask = paddle.ones(shape=input.shape)
        if is_fp16:
            input_mask = input_mask.cast(paddle.float32)
            grid = grid.cast(paddle.float32)
        output_mask = F.grid_sample(input_mask, grid)
        if is_fp16:
            output_mask = output_mask.cast(data_type)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output


# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    N = input_points.shape[0]
    M = control_points.shape[0]
    pairwise_diff = paddle.reshape(
        input_points, shape=[N, 1, 2]) - paddle.reshape(
            control_points, shape=[1, M, 2])
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :,
                                                                         1]
    repr_matrix = 0.5 * pairwise_dist * paddle.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = np.array(repr_matrix != repr_matrix)
    repr_matrix[mask] = 0
    return repr_matrix


# output_ctrl_pts are specified, according to our task.
def build_output_control_points(num_control_points, margins):
    margin_x, margin_y = margins
    num_ctrl_pts_per_side = num_control_points // 2
    ctrl_pts_x = np.linspace(margin_x, 1.0 - margin_x, num_ctrl_pts_per_side)
    ctrl_pts_y_top = np.ones(num_ctrl_pts_per_side) * margin_y
    ctrl_pts_y_bottom = np.ones(num_ctrl_pts_per_side) * (1.0 - margin_y)
    ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
    ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
    output_ctrl_pts_arr = np.concatenate(
        [ctrl_pts_top, ctrl_pts_bottom], axis=0)
    output_ctrl_pts = paddle.to_tensor(output_ctrl_pts_arr)
    return output_ctrl_pts


class TPSSpatialTransformer(nn.Layer):
    def __init__(self,
                 output_image_size=None,
                 num_control_points=None,
                 margins=None):
        super(TPSSpatialTransformer, self).__init__()
        self.output_image_size = output_image_size
        self.num_control_points = num_control_points
        self.margins = margins

        self.target_height, self.target_width = output_image_size
        target_control_points = build_output_control_points(num_control_points,
                                                            margins)
        N = num_control_points

        # create padded kernel matrix
        forward_kernel = paddle.zeros(shape=[N + 3, N + 3])
        target_control_partial_repr = compute_partial_repr(
            target_control_points, target_control_points)
        target_control_partial_repr = paddle.cast(target_control_partial_repr,
                                                  forward_kernel.dtype)
        forward_kernel[:N, :N] = target_control_partial_repr
        forward_kernel[:N, -3] = 1
        forward_kernel[-3, :N] = 1
        target_control_points = paddle.cast(target_control_points,
                                            forward_kernel.dtype)
        forward_kernel[:N, -2:] = target_control_points
        forward_kernel[-2:, :N] = paddle.transpose(
            target_control_points, perm=[1, 0])
        # compute inverse matrix
        inverse_kernel = paddle.inverse(forward_kernel)

        # create target cordinate matrix
        HW = self.target_height * self.target_width
        target_coordinate = list(
            itertools.product(
                range(self.target_height), range(self.target_width)))
        target_coordinate = paddle.to_tensor(target_coordinate)  # HW x 2
        Y, X = paddle.split(
            target_coordinate, target_coordinate.shape[1], axis=1)
        Y = Y / (self.target_height - 1)
        X = X / (self.target_width - 1)
        target_coordinate = paddle.concat(
            [X, Y], axis=1)  # convert from (y, x) to (x, y)
        target_coordinate_partial_repr = compute_partial_repr(
            target_coordinate, target_control_points)
        target_coordinate_repr = paddle.concat(
            [
                target_coordinate_partial_repr, paddle.ones(shape=[HW, 1]),
                target_coordinate
            ],
            axis=1)

        # register precomputed matrices
        self.inverse_kernel = inverse_kernel
        self.padding_matrix = paddle.zeros(shape=[3, 2])
        self.target_coordinate_repr = target_coordinate_repr
        self.target_control_points = target_control_points

    def forward(self, input, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.shape[1] == self.num_control_points
        assert source_control_points.shape[2] == 2
        batch_size = paddle.shape(source_control_points)[0]

        padding_matrix = paddle.expand(
            self.padding_matrix, shape=[batch_size, 3, 2])
        Y = paddle.concat([
            source_control_points.astype(padding_matrix.dtype), padding_matrix
        ], 1)
        mapping_matrix = paddle.matmul(self.inverse_kernel, Y)
        source_coordinate = paddle.matmul(self.target_coordinate_repr,
                                          mapping_matrix)

        grid = paddle.reshape(
            source_coordinate,
            shape=[-1, self.target_height, self.target_width, 2])
        grid = paddle.clip(grid, 0,
                           1)  # the source_control_points may be out of [0, 1].
        # the input to grid_sample is normalized [-1, 1], but what we get is [0, 1]
        grid = 2.0 * grid - 1.0
        output_maps = grid_sample(input, grid, canvas=None)
        return output_maps, source_coordinate