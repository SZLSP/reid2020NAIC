import os

cmdtext = """

python ./tools/submit.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-proto.yml --gpu-id 0 --test-iter 225599 --test-permutation
python ./tools/submit.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-ntl.yml --gpu-id 0 --test-iter 115199,119999,110399 --test-permutation
python ./tools/submit.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-iebn2.yml --gpu-id 0 --test-iter 225599 --test-permutation
"""
cmds = [cmd.strip().replace('train_net.py', 'submit.py') for cmd in cmdtext.split('\n')]
cmds = list(filter(lambda x: len(x) > 0, cmds))
for cmd in cmds:
    print(cmd)
    os.system(cmd)
