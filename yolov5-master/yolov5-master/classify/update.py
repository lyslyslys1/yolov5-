import torch.nn.functional as F


def mixup(data, targets, alpha=1.0):
    # 随机生成 mixup 权重
    weights = torch.FloatTensor(numpy.random.beta(alpha, alpha, size=(data.size(0), 1, 1, 1))).to(data.device)
    inverted_weights = 1.0 - weights

    # 对输入数据进行 mixup，同时对目标标签进行线性插值
    mixed_data = (data * weights) + (data.flip(0) * inverted_weights)
    mixed_targets = (targets * weights) + (targets.flip(0) * inverted_weights)

    return mixed_data, mixed_targets


class ImprovedYOLOv5(nn.Module):
    def __init__(self, num_classes, input_channels=3, anchors=None):
        super().__init__()
        # 其他参数的初始化

        if anchors is not None:
            self.anchor_boxes = [AnchorBox(x[0], x[1]) for x in anchors]
        else:
            self.anchor_boxes = None

        # mixup 增强的 alpha 参数
        self.alpha = 0.5

    def forward(self, x, targets=None):
        # 其他参数的 forward 实现

        if self.training:
            # 对于训练数据，直接使用源数据，不用 mixup
            if targets is not None or not self.is_mixup_enabled:
                loss, outputs = self.compute_loss(predictions, targets)
            # 对于训练数据，使用 mixup 数据增强
            else:
                mixed_x, mixed_targets = mixup(x, targets, self.alpha)
                mixed_prediction = self.predict(mixed_x)
                loss, outputs = self.compute_loss(mixed_prediction, mixed_targets)

        return loss, outputs

    def compute_loss(self, predictions, targets):
        # 其他参数的损失函数计算

        # 针对 mixup 数据增强，进行预测结果的解混合处理
        if self.is_mixup_enabled:
            predictions.flip(0)[:batch_size // 2] += predictions[len(predictions) // 2:].flip(0)[:batch_size // 2]
            predictions.flip(0)[:batch_size // 2] /= 2.0

        return loss, outputs