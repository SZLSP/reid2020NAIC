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
  --gpu-id 1

python ./tools/train_net.py \
  --config-file ./configs/HIT/sb-EFN5-arcF-cj-ranger-19.yml \
  --gpu-id 2

python ./tools/train_net.py \
  --config-file ./configs/HIT/sb-EFN5-arcF-cj-ranger-gray-19.yml \
  --gpu-id 3

# 2020.9.15

python ./tools/train_net.py \
  --config-file ./configs/HIT/sb-EFN5-arcF-gem-cj-ranger-19.yml \
  --gpu-id 0

python ./tools/train_net.py \
  --config-file ./configs/HIT/sb-EFN0-arcF-cj-19.yml \
  --gpu-id 2

python ./tools/train_net.py \
  --config-file ./configs/HIT/sb-EFN2-arcF-cj-19.yml \
  --gpu-id 4

python ./tools/train_net.py \
  --config-file ./configs/HIT/sb-EFN4-arcF-cj-19.yml \
  --gpu-id 5

# 2020.9.16

python ./tools/train_net.py \
  --config-file ./configs/HIT/sb-EFN5-arcF-cj-gray-19.yml \
  --gpu-id 0

python ./tools/train_net.py \
  --config-file ./configs/HIT/sb-EFN0-arcF-cj-19.yml \
  --gpu-id 1

python ./tools/train_net.py \
  --config-file ./configs/HIT/sb-EFN2-arcF-cj-19.yml \
  --gpu-id 2

python ./tools/train_net.py \
  --config-file ./configs/HIT/sb-EFN4-arcF-cj-19.yml \
  --gpu-id 3

python ./tools/train_net.py \
  --config-file configs/HIT/bt-EFN0-arcF-cj-19.yml \
  --gpu-id 4

python ./tools/train_net.py \
  --config-file configs/HIT/bt-EFN2-arcF-cj-19.yml \
  --gpu-id 5

python ./tools/train_net.py \
  --config-file configs/HIT/bt-EFN4-arcF-cj-19.yml \
  --gpu-id 6

python ./tools/train_net.py \
  --config-file configs/HIT/mgn-arcF-cj-19.yml \
  --gpu-id 7

python ./tools/train_net.py \
  --config-file configs/HIT/bt-R101-arcF-cj-gray-19.yml \
  --gpu-id 0

# 2020.9.17

python ./tools/train_net.py \
  --config-file configs/HIT/mgn-arcF-gempool-cj-19.yml \
  --gpu-id 7

# 2020.9.18

python ./tools/train_net.py \
  --config-file configs/HIT/bt-R101-arcF-cj-gray-19-500epoch.yml \
  --gpu-id 7


python ./tools/train_net.py \
  --config-file configs/HIT/bt-arcF-cj-ct-19.yml \
  --gpu-id 5

python ./tools/train_net.py \
  --config-file configs/HIT/bt-arcF-cj-19.yml \
  --gpu-id 2

# 2020.9.20

python ./tools/train_net.py \
  --config-file configs/HIT/R101-ibn-arcF-cj-st_in1-amp-gem-all-ranger2-gray.yml \
  --gpu-id 4

python ./tools/train_net.py \
  --config-file configs/HIT/R101-ibn-arcF-cj-st_in1-amp-gem-all-ranger2.yml \
  --gpu-id 5

python ./tools/train_net.py \
  --config-file configs/HIT/mgn-arcF-gempool-cj-19.yml \
  --gpu-id 7 --resume


# 2020.9.22

python ./tools/train_net.py \
  --config-file configs/HIT/mgn-arcF-cj-19.yml \
  --gpu-id 6 --resume

python ./tools/train_net.py \
  --config-file configs/HIT/R101-ibn-arcF-cj-st_in1-amp-gem-all-ranger2.yml \
  --gpu-id 5

python ./tools/submit.py --config-file ./configs/HIT/mgn-arcF-gempool-cj-19.yml --gpu-id 4 --test-permutation

python ./tools/train_net.py \
  --config-file configs/HIT/mgn-plain.yml \
  --gpu-id 5

python ./tools/train_net.py \
  --config-file configs/HIT/mgn-R101-plain.yml \
  --gpu-id 7

# 2020.09.24

python ./tools/submit.py --config-file configs/HIT/mgn-R101-plain.yml \
  --gpu-id 5 --test-permutation \
  --test-iter 0077999

python ./tools/train_net.py \
  --config-file configs/HIT/mgn--R101-cj.yml \
  --gpu-id 7

python ./tools/train_net.py \
  --config-file configs/HIT/mgn--R101-plain.yml \
  --gpu-id 5

python ./tools/submit.py --config-file configs/HIT/mgn-R101-plain.yml \
  --gpu-id 4 \
  --test-iter 0077999

python ./tools/submit.py --config-file configs/HIT/mgn-R101-plain.yml \
  --gpu-id 0 --test-permutation

# 2020.09.25

python ./tools/train_net.py \
  --config-file configs/HIT/mgn--R101-rea.yml \
  --gpu-id 7

# try resnetst

python ./tools/train_net.py \
  --config-file configs/HIT/mgn--resnetst-101-plain.yml \
  --gpu-id 3

python ./tools/train_net.py \
  --config-file configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-l2.yml \
  --gpu-id 1

python ./tools/train_net.py \
  --config-file configs/HIT/mgn--resnetst-50-plain.yml \
  --gpu-id 4

python ./tools/submit.py --config-file configs/HIT/mgn--R101-plain.yml \
  --gpu-id 0 --test-permutation

# 2020.9.26

python ./tools/train_net.py \
  --config-file configs/HIT/mgn--resnetst-101-plain.yml \
  --gpu-id 4

# 2020.9.27

python ./tools/train_net.py \
  --config-file configs/HIT/mgn--resnest-101-plain.yml \
  --gpu-id 7

python ./tools/train_net.py \
  --config-file configs/NAIC/R101-ibn-arcF-cj-st.yml \
  --gpu-id 5

# 2020.9.29

python ./tools/train_net.py \
  --config-file configs/HIT/mgn--R101-arcFace.yml \
  --gpu-id 7

python ./tools/train_net.py \
  --config-file configs/HIT/mgn--R101-0810.yml \
  --gpu-id 5

# 2020.10.4

python ./tools/submit.py --config-file configs/HIT/mgn--R101-0810.yml \
  --gpu-id 0 --test-permutation

python ./tools/submit.py --config-file configs/HIT/mgn--R101-arcFace.yml \
  --gpu-id 0 --test-permutation

python ./tools/train_net.py \
  --config-file configs/HIT/mgn--R101-arcFace-0710.yml \
  --gpu-id 5

# 2020.10.5

python ./tools/train_net.py \
  --config-file configs/HIT/mgn--R101-arcFace-0710.yml \
  --gpu-id 5

python ./tools/train_net.py \
  --config-file configs/HIT/mgn--R101-arcFace-gempool-0710.yml \
  --gpu-id 7


# 2020.10.11

python ./tools/submit.py --config-file configs/HIT/mgn--R101-rea-0810.yml \
  --gpu-id 0 --test-permutation

# 2020.10.25

python ./tools/submit.py --config-file configs/HIT/mgn--R101-plain.yml \
  --gpu-id 7 --test-permutation

# 2020.10.29

python ./tools/train_net.py --config-file configs/NAICREP/R101-ibn-arcF-cj-ca-st-amp-gem-all-ranger2-bs256-20dataonly.yml \
  --gpu-id 7

# 2020.10.30

python ./tools/submit.py --config-file configs/NAICREP/R101-ibn-arcF-cj-ca-st-amp-gem-all-ranger2-bs256-20dataonly.yml \
  --gpu-id 6 --test-permutation

# 2020.11.2

python ./tools/train_net.py --gpu-id 5 --config-file configs/NAICREP/mgn-R101-0910-AFF.yml
python ./tools/train_net.py --gpu-id 7 --config-file configs/NAICREP/mgn-R101-0910-iAFF.yml

# 2020.11.3

python ./tools/submit.py --config-file configs/NAICREP/mgn-R101-0910-AFF.yml \
  --gpu-id 5 --test-permutation
python ./tools/train_net.py --gpu-id 5 --config-file configs/NAICREP/mgn-R101-0910-AFF-allData.yml # out path unmodified

# 2020.11.4

python ./tools/submit.py --config-file configs/NAICREP/mgn-R101-0910-iAFF.yml \
  --gpu-id 0 --test-permutation

python ./tools/train_net.py --gpu-id 7 --config-file configs/NAICREP/mgn-R101-0910-iAFF-allData.yml

# 2020.11.8

python ./tools/submit.py --config-file configs/NAICREP/mgn-R101-0910-iAFF-allData.yml \
  --gpu-id 1 --test-permutation


# 2020.11.16

python ./tools/train_net.py --gpu-id 7 \
--config-file configs/NAICREP/R101-ibn-causal-cj-ca-st-amp-gem-all-ranger2-bs256-no19.yml


# 2020.11.19

python ./tools/train_net.py --gpu-id 7 \
--config-file configs/NAICREP/R50-bs256-no19.yml

# 2020.11.20

python ./tools/submit.py --config-file configs/NAICREP/R50-bs256-no19.yml \
  --gpu-id 7 --test-permutation


# 2020.11.23

python ./tools/submit.py --config-file configs/NAICREP/R101-ibn-arcF-cj-ca-st-amp-gem-all-ranger2-bs256-20dataonly.yml \
  --gpu-id 7 --test-permutation

python ./tools/submit.py --config-file configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-big.yml \
  --gpu-id 6 --test-permutation

  python ./tools/submit.py --config-file configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-269x.yml \
  --gpu-id 5 --test-permutation




# TEMPLATE

# Train
python ./tools/train_net.py --gpu-id 7 \
--config-file xxxxx

# Test
python ./tools/submit.py --config-file xxxxx \
  --gpu-id 7 --test-permutation

