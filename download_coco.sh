#! /usr/bin/bash

coco_url='http://images.cocodataset.org/zips/train2017.zip'
annotations_url='http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
download_location='/work3/s220493/coco/'

mkdir $download_location

rm $download_location/train2017.zip*
rm $download_location/annotations_trainval2017.zip*

wget -P $download_location $coco_url
wget -P $download_location $annotations_url

unzip $download_location/train2017.zip -d $download_location
unzip $download_location/annotations_trainval2017.zip -d $download_location

rm $download_location/train2017.zip*
rm $download_location/annotations_trainval2017.zip*

