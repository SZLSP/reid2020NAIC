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

python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp.yml --gpu-id 1 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-AA.yml --gpu-id 2 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-AM.yml --gpu-id 3 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem.yml --gpu-id 4 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-nl.yml --gpu-id 5 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-se.yml --gpu-id 6 --test-permutation

python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-bfb.yml --gpu-id 1 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-bgb.yml --gpu-id 2 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-bgn.yml --gpu-id 3 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-hfb.yml --gpu-id 4 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-hgb.yml --gpu-id 5 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-hgn.yml --gpu-id 6 --test-permutation

python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-ade.yml --gpu-id 1 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-coslr.yml --gpu-id 4 --test-permutation

python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-fl.yml --gpu-id 3 --test-permutation MODEL.WEIGHTS /home/dgy/project/pytorch/fast-reid/logs/NAIC/R101-ibn-arcF-cj-st-amp-gem/model_0045999.pth
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-smt1.yml --gpu-id 5 --test-permutation MODEL.WEIGHTS /home/dgy/project/pytorch/fast-reid/logs/NAIC/R101-ibn-arcF-cj-st-amp-gem/model_0045999.pth
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-smt2.yml --gpu-id 6 --test-permutation MODEL.WEIGHTS /home/dgy/project/pytorch/fast-reid/logs/NAIC/R101-ibn-arcF-cj-st-amp-gem/model_0045999.pth

python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-smt3.yml --gpu-id 2 --test-permutation MODEL.WEIGHTS /home/dgy/project/pytorch/fast-reid/logs/NAIC/R101-ibn-arcF-cj-st-amp-gem/model_0045999.pth
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-smt4.yml --gpu-id 3 --test-permutation MODEL.WEIGHTS /home/dgy/project/pytorch/fast-reid/logs/NAIC/R101-ibn-arcF-cj-st-amp-gem/model_0045999.pth
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-cl.yml --gpu-id 4 --test-permutation MODEL.WEIGHTS /home/dgy/project/pytorch/fast-reid/logs/NAIC/R101-ibn-arcF-cj-st-amp-gem/model_0045999.pth
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-smt5.yml --gpu-id 5 --test-permutation MODEL.WEIGHTS /home/dgy/project/pytorch/fast-reid/logs/NAIC/R101-ibn-arcF-cj-st-amp-gem/model_0045999.pth
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-smt6.yml --gpu-id 6 --test-permutation MODEL.WEIGHTS /home/dgy/project/pytorch/fast-reid/logs/NAIC/R101-ibn-arcF-cj-st-amp-gem/model_0045999.pth

python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2.yml --gpu-id 2 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger-swa.yml --gpu-id 3 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all.yml --gpu-id 4 --test-permutation MODEL.WEIGHTS /home/dgy/project/pytorch/fast-reid/logs/NAIC/R101-ibn-arcF-cj-st-amp-gem/model_0045999.pth
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all2.yml --gpu-id 5 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger.yml --gpu-id 6 --test-permutation

python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-l2.yml --gpu-id 2 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-l3.yml --gpu-id 3 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-l4.yml --gpu-id 4 --test-permutation

python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-big.yml --gpu-id 1 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-200x.yml --gpu-id 2 --test-permutation

python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-cent.yml --gpu-id 1 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-l3.yml --gpu-id 3 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-l4.yml --gpu-id 4 --test-permutation

python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-osnet75-amp-gem-all-ranger2.yml --gpu-id 5 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-osnetx1_0-amp-gem-all-ranger2.yml --gpu-id 6 --test-permutation

python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-adabn.yml --gpu-id 1 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-attbn.yml --gpu-id 2 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-brn.yml --gpu-id 3 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-IEbn.yml --gpu-id 4 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-sbn.yml --gpu-id 5 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-rea.yml --gpu-id 6 --test-permutation

python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-1s4x24d.yml --gpu-id 1 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-4s1x64d.yml --gpu-id 2 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-2s2x40d.yml --gpu-id 3 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-iebn2.yml --gpu-id 4 --test-permutation

python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-4s2x40d.yml --gpu-id 5 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-ntl.yml --gpu-id 6 --test-permutation

python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st_in1-amp-gem-all-ranger2.yml --gpu-id 1 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st_in2-amp-gem-all-ranger2.yml --gpu-id 2 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st_in3-amp-gem-all-ranger2.yml --gpu-id 3 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st_in4-amp-gem-all-ranger2.yml --gpu-id 5 --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-in-arcF-cj-st-amp-gem-all-ranger2.yml --gpu-id 6 --test-permutation
