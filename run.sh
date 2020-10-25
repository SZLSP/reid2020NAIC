python ./tools/train_net.py \
  --config-file ./configs/Market1501/bagtricks_R50.yml \
  MODEL.DEVICE "cuda:0"

python ./tools/train_net.py \
  --config-file ./configs/HIT/R101-ibn-cS-cj-efn5-v2.yml \
  MODEL.DEVICE "cuda:5"

python ./tools/train_net.py \
  --config-file ./configs/HIT/sb-efn-cS-cj-efn5.yml \
  MODEL.DEVICE "cuda:2"


python ./tools/train_net.py \
  --config-file ./configs/HIT/R101-ibn-cS-cj-efn7.yml \
  MODEL.DEVICE "cuda:2"

python ./tools/train_net.py \
  --config-file ./configs/HIT/R101-ibn-cS-cj-efnl2.yml \
  MODEL.DEVICE "cuda:3"

python ./tools/submit.py --config-file ./configs/HIT/bagtricks_R101-ibn-big-cS_test.yml --gpu-id 4



python ./tools/train_net.py \
  --config-file ./configs/HIT/sb-EFN5-arcF-cj-19.yml \
  --gpu-id 7

python ./tools/train_net.py \
  --config-file ./configs/HIT/R101-ibn-arcF-cj-st-amp-gem-all-269x.yml \
  --gpu-id 4

