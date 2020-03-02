'''Create train and test files to be parsed through dataset.py and used
to train and evaluate the autoencoder.
'''

import csv
import os


def main():
    with open('cxr-record-list.csv') as f:
        reader = csv.reader(f, delimiter=',')
        headers = reader.__next__()
        paths = []
        for i, row in enumerate(reader):
            paths.append(row[3])
            if i == 1100:
                break
    print('\n'.join(paths[:1000]), file=open('train.csv', 'w'))
    print('\n'.join(paths[1000:1100]), file=open('test.csv', 'w'))


if __name__ == '__main__':
    main()
