import os

results = os.listdir('result')
imgs = os.listdir('test_imgs')

for img in imgs:
    if img in results:
        os.remove(os.path.join('test_imgs', img))