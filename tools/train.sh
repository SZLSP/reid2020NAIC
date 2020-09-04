python ./tools/train_net.py --config-file ./configs/NAIC/bagtricks_R50.yml --gpu-id 0
python ./tools/train_net.py --config-file ./configs/NAIC/bagtricks_R101-ibn.yml --gpu-id 0
python ./tools/train_net.py --config-file ./configs/NAIC/AGW_R101-ibn.yml --gpu-id 1
python ./tools/train_net.py --config-file ./configs/NAIC/sbs_R101-ibn.yml --gpu-id 2
python ./tools/train_net.py --config-file ./configs/NAIC/bagtricks_R101-ibn-ms.yml --gpu-id 3

python ./tools/train_net.py --config-file ./configs/NAIC/bagtricks_R101-ibn-ct.yml --gpu-id 0
python ./tools/train_net.py --config-file ./configs/NAIC/strong_R50-ibn-ct.yml --gpu-id 1
python ./tools/train_net.py --config-file ./configs/NAIC/strong_R50-ibn-ct-swa.yml --gpu-id 2

python ./tools/train_net.py --config-file ./configs/NAIC/strong_R50-ibn-swa.yml --gpu-id 0 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/strong_R50-ibn-swa-cl.yml --gpu-id 1 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/strong_R50-ibn-swa-fl.yml --gpu-id 2 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/strong_R50-ibn-swa-fl-cl.yml --gpu-id 3 --test-permutation

python ./tools/train_net.py --config-file ./configs/NAIC/bagtricks_R101-ibn-i8.yml --gpu-id 1 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/bagtricks_R101-ibn-i8-05.yml --gpu-id 2 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/bagtricks_R101-ibn-i8-da.yml --gpu-id 3 --test-permutation