model = dict(
    type='CT',
    backbone=dict(
        type='resnet18',
        pretrained=True
    ),
    neck=dict(
        type='FPEM_v1',
        in_channels=(64, 128, 256, 512),
        out_channels=128
    ),
    detection_head=dict(
        type='CT_Head',
        in_channels=512,
        hidden_dim=128,
        num_classes=3,
        loss_kernel=dict(
            type='DiceLoss',
            loss_weight=1.0
        ),
        loss_loc=dict(
            type='SmoothL1Loss',
            beta=0.1,
            loss_weight=0.05
        )
    )
)
data = dict(
    batch_size=16,
    train=dict(
        type='CT_MSRA',
        split='train',
        is_transform=True,
        img_size=736,
        short_size=736,
        kernel_scale=0.6,
        read_type='cv2'
    ),
    test=dict(
        type='CT_MSRA',
        split='test',
        short_size=736,
        read_type='cv2'
    )
)
train_cfg = dict(
    lr=1e-3,
    schedule='polylr',
    epoch=600,
    optimizer='Adam'
)
test_cfg = dict(
    min_score=0.86,
    min_area=16,
    bbox_type='rect',
    result_path='outputs/submit_msra/'
)
