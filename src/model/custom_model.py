import torch
import torch.nn as nn
import timm


class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_map = self.conv(x)
        attention_map = self.sigmoid(attention_map)  # Normalize to range [0, 1]
        return x * attention_map


class CustomModel(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool="",
            in_chans=1,
        )

        # Remove the original classification head of the backbone
        self.backbone.classifier = nn.Identity()

        self.attention = SpatialAttentionModule(self.backbone.num_features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.attention(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.head(x)
        return x


def get_timm_backbone(config, pretrained):
    backbone = timm.create_model(
        config.model.backbone_type,
        pretrained=pretrained,
        num_classes=0,
        global_pool="",
        in_chans=1,
    )
    return backbone
