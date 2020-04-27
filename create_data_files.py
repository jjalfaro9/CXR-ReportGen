import glob
import os

def main():
    data_path = "/data/mimic-cxr/files/"
    subfolders = ['p10', 'p11']

    if not os.path.isdir('./data'):
        os.system("mkdir data")
    else:
        os.system('rm data/*')

    img_file = open('data/all_images.txt', 'w')
    report_file = open('data/all_reports.txt', 'w')

    for sub in subfolders:
        for sub_sub in os.listdir(data_path + sub + '/'):
            try:
                for f in os.listdir(data_path + sub + '/' + sub_sub + '/'):
                    if '.txt' in f or '.html' in f:
                        pass
                    else:
                        for dcm_name in os.listdir(data_path + sub + '/' + sub_sub + '/' + f):
                            if '.html' in dcm_name:
                                continue

                            print('writing')
                            report_file.write(data_path + sub + '/' + sub_sub + '/' + f + '.txt\n')
                            img_file.write(data_path + sub + '/' + sub_sub + '/' + f + '/' + dcm_name + '\n')
            except NotADirectoryError:
                continue

main()



