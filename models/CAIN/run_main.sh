CUDA_VISIBLE_DEVICES=0 python main.py --exp_name VEC_TEST_INT --batch_size 4 --test_batch_size 1 --dataset anime_vectorized --model cain --max_epoch 200 --lr 0.0001 --data_root /media/sahil/DL_5TB/MachineLearning/anime_playlist_downloads/fight_scenes_extracted_frames --test_data_root /media/sahil/DL_5TB/MachineLearning/anime_playlist_downloads/test_fight_scenes --svg_dir /media/sahil/DL_5TB/MachineLearning/anime_playlist_downloads/fight_scenes_extracted_frames --test_svg_dir /media/sahil/DL_5TB/MachineLearning/anime_playlist_downloads/test_fight_scenes/