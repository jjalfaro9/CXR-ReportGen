import random

images = []
for line in open('data/all_images.txt'):
    images.append(line.strip())

reports = []
for line in open('data/all_reports.txt'):
    reports.append(line.strip())

num_files = len(images)
num_split = int(.2 * num_files)
split = random.sample(list(range(num_files)), num_split)

train_imgs = open('data/train_images.txt', 'w')
train_reps = open('data/train_reports.txt', 'w')
test_imgs = open('data/test_images.txt', 'w')
test_reps = open('data/test_reports.txt', 'w')

zip_i_r = list(zip(images, reports))
for i in range(len(zip_i_r)):
	img, rep = zip_i_r[i]
	if i in split:
		print('split')
		test_imgs.write(img+'\n')
		test_reps.write(rep+'\n')
	else:
		train_imgs.write(img+'\n')
		train_reps.write(rep+'\n')


