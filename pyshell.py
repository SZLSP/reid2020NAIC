import os

cmdtext = """
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger-swa.yml --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all.yml --test-permutation
python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all2.yml --test-permutation
"""
cmds = [cmd.strip().replace('train_net.py', 'submit.py') for cmd in cmdtext.split('\n')]
cmds = list(filter(lambda x: len(x) > 0, cmds))
for cmd in cmds:
    print(cmd)
    os.system(cmd)
