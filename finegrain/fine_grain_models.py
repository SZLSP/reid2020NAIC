"""
Bootstrap Your Own Latent (BYOL), in Pytorch
https://github.com/lucidrains/byol-pytorch

usage: python fine_grain_model.py --config-file path_to_config_gile
the script will output the path to fine-grained model
which can be used in the config setting, by key word "cfg.MODEL.BACKBONE.PRETRAIN_PATH"
"""

import torch
from byol_pytorch import BYOL
from torchvision import models

from finegrain.load_backbone import load_model, load_cfg
from fastreid.data import build_reid_test_loader, build_reid_train_loader

from tqdm import tqdm

cfg, model = load_model()
# cfg = load_cfg()

cfg.defrost()
cfg.SOLVER.IMS_PER_BATCH = 20
cfg.freeze()

train_loader = build_reid_train_loader(cfg)

DEVICE = torch.device(cfg.MODEL.DEVICE)

# print(model)
# model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True).to(DEVICE)
# model = models.resnet101(pretrained=True).to(DEVICE)

learner = BYOL(
    model,
    image_size=(256, 128),
    hidden_layer=3  # layer used to extract features
).to(DEVICE)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)


def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 128).to(DEVICE)


# for _ in range(100):
#     images = sample_unlabelled_images()
#     loss = learner(images)
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
#     learner.update_moving_average() # update moving average of target encoder

# print("total iter: ", len(train_loader))

count = 0
for data in tqdm(train_loader):
# for _ in range(10):
#     data = iter(train_loader)
    # images = sample_unlabelled_images()
    images = data['images'].to(DEVICE)
    loss = learner(images)
    opt.zero_grad()
    loss.backward()
    opt.step()
    learner.update_moving_average()  # update moving average of target encoder
    count += 1
    if count == 5000 : break

buid_method = cfg.MODEL.BACKBONE.NAME
model_name = buid_method.split("_")[1]
# save your improved network
torch.save(model.state_dict(), './finegrain/improved-{}.pth'.format(model_name))
print("model saved")
