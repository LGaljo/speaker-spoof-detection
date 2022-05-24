import os
import csv
import shutil

sample_dir = 'DS_10283_3055'
metadata_path = 'DS_10283_3055/metadata.csv'

if __name__ == '__main__':
    header = ['primary_label', 'secondary_labels', 'type', 'filename', 'split']
    f = open(metadata_path, 'w', encoding='UTF8', newline='')
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(header)

    for root, dirs, files in os.walk(sample_dir, topdown=False):
        if root.startswith('DS_10283_3055\\ASVspoof2017_V2'):
            # print(dirs)
            # print(files)
            print(root)

            dest = os.path.join(*[sample_dir, "test" if root.split("_")[-1] == "dev" else "train"])
            if not os.path.exists(dest):
                os.mkdir(os.path.join(dest))
                os.mkdir(os.path.join(*[dest, "spoof"]))
                os.mkdir(os.path.join(*[dest, "genuine"]))

            for filename in files:
                row = [
                    'spoof' if filename.startswith('spoof') else 'genuine',
                    [],
                    ['speech'],
                    filename,
                    "test" if root.split("_")[-1] == "dev" else "train"
                ]
                # shutil.copy2(os.path.join(root, filename), os.path.join(*[sample_dir, row[2], row[0], filename]))
                # print(row)
                writer.writerow(row)

    f.close()
