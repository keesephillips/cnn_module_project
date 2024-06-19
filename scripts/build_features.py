import splitfolders
import os

if __name__ == '__main__':
    for subdir, dirs, files in os.walk(os.getcwd() + '/../data/raw'):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".jpg"):
                
                folder = filepath.split("\\")[-2].split("-")[1]

                if not os.path.isdir(os.getcwd() + "/../data/raw/images_folder/" + folder): 
                    os.makedirs(os.getcwd() + "/../data/raw/images_folder/" + folder) 
                
                os.rename(filepath, os.getcwd() + "/../data/raw/images_folder/" + folder + "/" + filepath.split("\\")[-1])
                
    splitfolders.ratio('../data/raw/images_folder', output='../data/processed', seed=1337, ratio=(.8, 0.0,0.2)) 
