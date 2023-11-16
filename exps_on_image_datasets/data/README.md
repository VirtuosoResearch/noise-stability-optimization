### Data Preparation

We use several other datasets in our experiments. We list the link for downloading the datasets below:

- [Aircrafts](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/): download and extract into `./data/aircrafts`
  - remove the class `257.clutter` out of the data directory
- [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html): download and extract into `./data/CUB_200_2011/`
- [Caltech-256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/): download and extract into `./data/caltech256/`
- [Stanford-Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html): download and extract into `./data/StanfordCars/`
- [Stanford-Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/): download and extract into `./data/StanfordDogs/`
- [Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/): download and extract into `./data/flowers/`
- [MIT-Indoor](http://web.mit.edu/torralba/www/indoor.html): download and extract into `./data/Indoor/`

Our code automatically handles the processing of the datasets. 

### Messidor-2

The Messidor 2 dataset is an ophthalmology dataset, grading diabetic retinopathy on the 0-4 Davis Scale, with 4 being the most severe grading.

1. Navigate to the [Messidor-2 Database Download Page](https://www.adcis.net/en/third-party/messidor2/). Complete the license agreement, and a code will be emailed to you to use when downloading the dataset.
2. The dataset comes in a 4-part Zip archive. Create a folder titled `messidor2` in the `data_root/opthamology/` directory. Extract the multi-part archive into into an `IMAGES` folder under `messidor2`. The additional "Pairs left eye / right eye" csv file is optional for you to download, since it is not necessary for dataloading.
3. Navigate to the [Messidor2 Kaggle Link](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades) and download `messidor_data.csv` and `messidor_readme.txt` into the `messidor2` folder. The directory should now be structured as follows:

```
<data/fundus/messidor2>
├── messidor_data.csv
├── messidor_readme.txt
└── IMAGES
    ├── 20051020_43808_0100_PP.png
    ├── ...
    ├── IM004832.JPG
```

### APTOS 2019 Blindness Detection dataset

The APTOS 2019 Blindness Detection dataset grades diabetic retinopathy on the 0-4 Davis Scale from retina images taken using fundus photography, with 4 being the most severe grading.

1. Navigate to the [APTOS Kaggle link](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data). Under the "Rules" tab, accept the rules for the competition to download the dataset.
2. After downloading to a folder titled `aptos`, the directory should now be structured as follows:

```
<data/fundus/aptos>
├── sample_submission.csv
├── test.csv
├── train.csv
└── test_images
    ├── 0005cfc8afb6.png
    ├── ...
└── train_images
    ├── 000c1434d8d7.png
```
