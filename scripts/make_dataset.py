import urllib.request
import tarfile
import os

if __name__ == '__main__':
    # Download the data
    if not os.path.exists(os.getcwd() + '/data'):
        os.mkdir(os.getcwd() + '/data')
    if not os.path.exists(os.getcwd() + '/data/raw'):
        url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
        urllib.request.urlretrieve(url,filename='data/dogs.tar')
        # Open the tar file
        with tarfile.open(os.getcwd() + '/data/dogs.tar', 'r') as tar:
            # Extract all files to the 'data/raw' directory
            tar.extractall(path=os.getcwd() + '/data/raw')
            
    print("Dataset Collected... see data/")