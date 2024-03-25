# Prepare Datasets for PCM

## ScanObjectNN
### Download
```commandline
mkdir -p data/ScanObjectNN
cd data/ScanObjectNN
wget http://hkust-vgd.ust.hk/scanobjectnn/h5_files.zip
```
or 
```commandline
mkdir -p data/ScanObjectNN
cd data/ScanObjectNN
gdown https://drive.google.com/uc?id=1iM3mhMJ_N0x5pytcP831l3ZFwbLmbwzi
tar -xvf ScanObjectNN.tar
```
### Structure
Organize the dataset as follows:
```commandline
data
 |--- ScanObjectNN
            |--- h5_files
                    |--- main_split
                            |--- training_objectdataset_augmentedrot_scale75.h5
                            |--- test_objectdataset_augmentedrot_scale75.h5
```

## ModelNet40
### Download
```commandline
mkdir -p data/ModelNet40Ply2048
cd data/ModelNet40Ply2048
wget https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
```
### Structure
Organize the dataset as follows:
```commandline
data
 |--- ModelNet40Ply2048
            |--- modelnet40_ply_hdf5_2048
```

## ShapeNetPart
### Download
```commandline
cd data && mkdir ShapeNetPart && cd ShapeNetPart
gdown https://drive.google.com/uc?id=1W3SEE-dY1sxvlECcOwWSDYemwHEUbJIS
tar -xvf shapenetcore_partanno_segmentation_benchmark_v0_normal.tar
```
### Structure
Organize the dataset as follows:
```commandline
data
 |--- ShapeNetPart
        |--- shapenetcore_partanno_segmentation_benchmark_v0_normal
                |--- train_test_split
                      |--- shuffled_train_file_list.json
                      |--- ...
                |--- 02691156
                      |--- 1a04e3eab45ca15dd86060f189eb133.txt
                      |--- ...               
                |--- 02773838
                |--- synsetoffset2category.txt
                |--- processed
                        |--- trainval_2048_fps.pkl
                        |--- test_2048_fps.pkl
```

## S3DIS
### Download
```commandline
mkdir -p data/S3DIS/
cd data/S3DIS
gdown https://drive.google.com/uc?id=1MX3ZCnwqyRztG1vFRiHkKTz68ZJeHS4Y
tar -xvf s3disfull.tar
```
### Structure
Organize the dataset as follows:
```commandline
data
 |--- S3DIS
        |--- s3disfull
                |--- raw
                      |--- Area_6_pantry_1.npy
                      |--- ...
                |--- processed
                      |--- s3dis_val_area5_0.040.pkl 
```