import json
import pickle
import os
import re

class Indexer(object):
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        if (index not in self.ints_to_objs):
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        return self.index_of(object) != -1

    # Returns -1 if the object isn't present, index otherwise
    def index_of(self, object):
        if (object not in self.objs_to_ints):
            return -1
        else:
            return self.objs_to_ints[object]

    # Adds the object to the index if it isn't present, always returns a nonnegative index
    def get_index(self, object, add=True):
        if not add:
            return self.index_of(object)
        if (object not in self.objs_to_ints):
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]


def addWordsToIndexer(file, idxr):
    report_path = '../png_files_sample/label/{name}.txt'.format(name=file)
    with open (report_path, "r") as r_file:
        file_read = r_file.read()
        report = re.split("[\n:]", file_read)
        for i in range(len(report)):
            report[i] = report[i].strip().lower()

    try:
        index = report.index('findings')
    except ValueError:
        index = report.index('findings and impression')
    try:
        index2 = report.index('impression')
    except ValueError:
        index2 = len(report)

    sentences = ' '.join(report[index+2:index2]).split('. ')

    target = []
    for i, sentence in enumerate(sentences):
        sentence = sentence.lower().replace('.', '').replace(',', '').split()
        if len(sentence) == 0: # or len(sentence) > self.n_max:
            continue
        
        for token in sentence:
            idxr.get_index(token)

if __name__ == '__main__':
    idxr = Indexer()
    idxr.get_index('<start>')
    idxr.get_index('<end>')
    idxr.get_index('<unk>')

    skipped = 0
    files = []

    for file in os.listdir('../png_files_sample/label'):
        files.append(file[:-4])

    for f in files:
        try:
            addWordsToIndexer(f, idxr)
        except ValueError:
            skipped += 1


    print(skipped)
    print(len(files))

    # print(idxr.objs_to_ints.keys())

    pickle.dump(idxr.objs_to_ints, open('sample_idxr-obj', 'wb'))  
   






