import os
import wave


def checkwav(filename):
    with wave.open(os.path.join(*[root, filename]), 'r') as fin:
        header_fsize = (fin.getnframes() * fin.getnchannels() * fin.getsampwidth()) + 44
    file_fsize = os.path.getsize(os.path.join(*[root, filename]))
    print(filename, header_fsize != file_fsize)


if __name__ == '__main__':
    for root, dirs, filenames in os.walk('owndataset', topdown=False):
        for filename in filenames:
            checkwav(filename)
