python ./tools/submit.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st.yml --gpu-id 0 --test-sets NAIC19Test --test-permutation MODEL.WEIGHTS logs/NAIC/R101-ibn-arcF-cj-st/model_0068999.pth
python ./tools/submit.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st.yml --gpu-id 0 --test-sets NAIC19Test --test-permutation MODEL.WEIGHTS logs/NAIC/R101-ibn-arcF-cj-st/model_0064399.pth
python ./tools/submit.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st.yml --gpu-id 0 --test-sets NAIC19Test --test-permutation MODEL.WEIGHTS logs/NAIC/R101-ibn-arcF-cj-st/model_0055199.pth
python ./tools/submit.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st.yml --gpu-id 0 --test-sets NAIC19Test --test-permutation MODEL.WEIGHTS logs/NAIC/R101-ibn-arcF-cj-st/model_0041399.pth

python ./tools/submit.py --config-file ./configs/NAIC/R101-ibn-adaC-cj-st.yml --gpu-id 0 --test-sets NAIC19Test --test-permutation MODEL.WEIGHTS logs/NAIC/R101-ibn-adaC-cj-st/model_0059799.pth
python ./tools/submit.py --config-file ./configs/NAIC/R101-ibn-adaC-cj-st.yml --gpu-id 0 --test-sets NAIC19Test --test-permutation MODEL.WEIGHTS logs/NAIC/R101-ibn-adaC-cj-st/model_0045999.pth
python ./tools/submit.py --config-file ./configs/NAIC/R101-ibn-adaC-cj-st.yml --gpu-id 0 --test-sets NAIC19Test --test-permutation MODEL.WEIGHTS logs/NAIC/R101-ibn-adaC-cj-st/model_0041399.pth

python ./tools/submit.py --config-file ./configs/NAIC/R101-ibn-arcC-cj-st.yml --gpu-id 0 --test-sets NAIC19Test --test-permutation MODEL.WEIGHTS logs/NAIC/R101-ibn-arcC-cj-st/model_0059799.pth
python ./tools/submit.py --config-file ./configs/NAIC/R101-ibn-arcC-cj-st.yml --gpu-id 0 --test-sets NAIC19Test --test-permutation MODEL.WEIGHTS logs/NAIC/R101-ibn-arcC-cj-st/model_0055199.pth
python ./tools/submit.py --config-file ./configs/NAIC/R101-ibn-arcC-cj-st.yml --gpu-id 0 --test-sets NAIC19Test --test-permutation MODEL.WEIGHTS logs/NAIC/R101-ibn-arcC-cj-st/model_0045999.pth
python ./tools/submit.py --config-file ./configs/NAIC/R101-ibn-arcC-cj-st.yml --gpu-id 0 --test-sets NAIC19Test --test-permutation MODEL.WEIGHTS logs/NAIC/R101-ibn-arcC-cj-st/model_0036799.pth

python ./tools/submit.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem.yml --gpu-id 0 --test-sets NAIC19_REP --test-permutation
python ./tools/submit.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all.yml --gpu-id 0 --test-sets NAIC19_REP --test-permutation

python ./tools/submit.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all.yml --gpu-id 1 --test-iter 225599 --test-permutation

python ./tools/submit.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-proto.yml --gpu-id 0 --test-iter 225599 --test-permutation
python ./tools/submit.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-ntl.yml --gpu-id 0 --test-iter 115199,119999,110399 --test-permutation
python ./tools/submit.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2.iebn2 --gpu-id 0 --test-iter 225599 --test-permutation
