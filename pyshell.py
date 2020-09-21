import os

cmdtext = """

python ./tools/train_net.py --config-file ./configs/NAIC/R101-ibn-arcF-cj-st-amp-gem-all-ranger2-cent3.yml --gpu-id 0 --test-permutation
"""
cmds = [cmd.strip().replace('train_net.py', 'submit.py') for cmd in cmdtext.split('\n')]
cmds = list(filter(lambda x: len(x) > 0, cmds))
for cmd in cmds:
    print(cmd)
    os.system(cmd)
