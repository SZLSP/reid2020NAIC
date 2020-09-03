python ./tools/train_net.py --config-file ./configs/NAIC/bagtricks_R50.yml --gpu-id 0
python ./tools/train_net.py --config-file ./configs/NAIC/bagtricks_R101-ibn.yml --gpu-id 0
python ./tools/train_net.py --config-file ./configs/NAIC/AGW_R101-ibn.yml --gpu-id 1
python ./tools/train_net.py --config-file ./configs/NAIC/sbs_R101-ibn.yml --gpu-id 2
python ./tools/train_net.py --config-file ./configs/NAIC/bagtricks_R101-ibn-ms.yml --gpu-id 3