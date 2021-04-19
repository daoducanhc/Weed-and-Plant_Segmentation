import os

folders = ['mask', 'nir', 'rgb']

for folder in folders:
    for ind, filename in enumerate(os.listdir(folder)):
        old_name = os.path.join(folder, filename)
        new_name = os.path.join(folder, str(ind) + '.png')

        os.rename(old_name, new_name)
