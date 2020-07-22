import numpy as np

def load_ids(name, count=None, randomize=True):
    with open('data/scene_ids/{}.txt'.format(name)) as f:
        scene_ids = f.read().strip().split('\n')
        # Discard and '# comments'
        scene_ids = [x.split('#')[0].strip() for x in scene_ids]
        # And any blank lines
        scene_ids = [x for x in scene_ids if x]
    if randomize:
        np.random.seed(88)
        np.random.shuffle(scene_ids)
    if count is not None:
        scene_ids = scene_ids[:count]
    return scene_ids    