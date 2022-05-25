import os

path = 'DS_10283_3055/protocol_V2/ASVspoof2017_V2_eval.trl.txt'
sample_dir = 'DS_10283_3055/ASVspoof2017_V2_eval'

labels_data = dict()


def rename_files(f):
    for line in f.readlines():
        chunks = line.split(' ')
        new_filename = chunks[1] + '_' + chunks[0]
        print(new_filename)
        old_file = os.path.join(sample_dir, chunks[0])
        new_file = os.path.join(sample_dir, new_filename)
        # print(old_file, os.path.exists(old_file))
        # print(new_file, os.path.exists(new_file))
        os.rename(old_file, new_file)


if __name__ == '__main__':
    if os.path.exists(path):
        file = open(path)
        rename_files(file)
        file.close()
