ROOT_PATH=$1
DATASET_PATH=$1/UCF101
ORIGIN_PATH=$DATASET_PATH/UCF101
CLASSIFY_PATH=$DATASET_PATH/videos_classified
VIDEO_PATH=$DATASET_PATH/videos_jpeg
ANNO_PATH=$DATASET_PATH/annotations

chmod u+x scripts/*.sh
mkdir $DATASET_PATH

scripts/download_dataset.sh $DATASET_PATH
scripts/download_annotations.sh $DATASET_PATH

python3 utils/classify_video.py $ORIGIN_PATH $CLASSIFY_PATH 
python3 utils/video_jpg_ucf101_hmdb51.py $CLASSIFY_PATH $VIDEO_PATH
python3 utils/n_frames_ucf101_hmdb51.py $VIDEO_PATH 

# using only one class
echo "using only one class: PlayingViolin"
mv $ANNO_PATH/classInd.txt $ANNO_PATH/classInd.txt.bak
echo "1 PlayingViolin" >> $ANNO_PATH/classInd.txt

python3 utils/ucf101_json.py $ANNO_PATH

rm -rf $ORIGIN_PATH $CLASSIFY_PATH
