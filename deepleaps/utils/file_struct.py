import os, shutil
from collections import defaultdict

def get_meta_value(workspace_meta, task, name, id):
    try:
        meta: dict = workspace_meta[task][name]
        if id == '$latest':
            id = max(meta.keys())
    except Exception:
        return None
    return meta[id]

def get_workspace_meta(base):
    out = defaultdict(lambda:defaultdict(lambda:defaultdict(None)))

    if os.path.isfile(os.path.join(base, 'ckp.meta')):
        with open(os.path.join(base, 'ckp.meta')) as f:
            for line in list(map(lambda x: x.strip(), f.readlines())):
                line: str
                task, module_name, id, data = line.split(' ', maxsplit=3)
                out[task][module_name][id] = data
    return out

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
