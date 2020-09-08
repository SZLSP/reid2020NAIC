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

python ./tools/train_net.py --config-file ./configs/NAIC/bagtricks_R101-ibn-cj.yml --gpu-id 0 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/bagtricks_R101-ibn-cj-se.yml --gpu-id 1 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/bagtricks_R101-ibn-amS.yml --gpu-id 2 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/bagtricks_R101-ibn-aS.yml --gpu-id 3 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/bagtricks_R101-ibn-gem.yml --gpu-id 4 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/bagtricks_R101-ibn-cS.yml --gpu-id 5 --test-permutation

python ./tools/train_net.py --config-file ./configs/NAIC/mgn_R50-ibn.yml --gpu-id 2 --test-permutation

python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-cS-cj-amp.yml --gpu-id 0 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-cS-cj-fap.yml --gpu-id 1 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-cS-cj-id.yml --gpu-id 2 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-cS-cj-st.yml --gpu-id 3 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-cS-cj-xt.yml --gpu-id 4 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-cS-cj-xt-gbn.yml --gpu-id 5 --test-permutation

python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-cS-cj-efn0.yml --gpu-id 1 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-cS-cj-efn3.yml --gpu-id 2 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-cS-cj-efn5.yml --gpu-id 5 --test-permutation

python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-adaC-cj-st.yml --gpu-id 1 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcC-cj-st.yml --gpu-id 2 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st.yml --gpu-id 3 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-cosF-cj-st.yml --gpu-id 4 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-cS-cj-st-gem.yml --gpu-id 5 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-sphF-cj-st.yml --gpu-id 6 --test-permutation

