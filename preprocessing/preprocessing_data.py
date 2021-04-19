import os

file_names = set(os.listdir('mask'))

for f in os.listdir('nir'):
    if f not in file_names:
        os.remove(os.path.join('nir', f))