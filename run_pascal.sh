#Zero Label
CUDA_VISIBLE_DEVICES=$1 python train.py --config config/voc12/ZLSS.yaml --experimentid myexp --imagedataset voc12
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/voc12/ZLSS.yaml --imagedataset voc12 --model-path logs/voc12/myexp/checkpoint_20000.pth.tar -r zlss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/voc12/ZLSS.yaml --imagedataset voc12 --model-path logs/voc12/myexp/checkpoint_20000.pth.tar -r gzlss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/voc12/ZLSS.yaml --imagedataset voc12 --model-path logs/voc12/myexp/checkpoint_20000.pth.tar -r gzlss --threshold 0.6

#Few Label
CUDA_VISIBLE_DEVICES=$1 python train.py --config config/voc12/FLSS.yaml --experimentid myexp --continue-from 20000 --nshot 1 --imagedataset voc12 --inputmix both
CUDA_VISIBLE_DEVICES=$1 python train.py --config config/voc12/FLSS.yaml --experimentid myexp --continue-from 20000 --nshot 2 --imagedataset voc12 --inputmix both
CUDA_VISIBLE_DEVICES=$1 python train.py --config config/voc12/FLSS.yaml --experimentid myexp --continue-from 20000 --nshot 5 --imagedataset voc12 --inputmix both
CUDA_VISIBLE_DEVICES=$1 python train.py --config config/voc12/FLSS.yaml --experimentid myexp --continue-from 20000 --nshot 10 --imagedataset voc12 --inputmix both
CUDA_VISIBLE_DEVICES=$1 python train.py --config config/voc12/FLSS.yaml --experimentid myexp --continue-from 20000 --nshot 20 --imagedataset voc12 --inputmix both
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/voc12/FLSS.yaml --imagedataset voc12 --model-path logs/voc12/myexp/checkpoint_20000_5b_0_2000.pth.tar -r flss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/voc12/FLSS.yaml --imagedataset voc12 --model-path logs/voc12/myexp/checkpoint_20000_5b_0_2000.pth.tar -r gflss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/voc12/FLSS.yaml --imagedataset voc12 --model-path logs/voc12/myexp/checkpoint_20000_10b_0_2000.pth.tar -r flss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/voc12/FLSS.yaml --imagedataset voc12 --model-path logs/voc12/myexp/checkpoint_20000_10b_0_2000.pth.tar -r gflss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/voc12/FLSS.yaml --imagedataset voc12 --model-path logs/voc12/myexp/checkpoint_20000_1b_0_2000.pth.tar -r flss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/voc12/FLSS.yaml --imagedataset voc12 --model-path logs/voc12/myexp/checkpoint_20000_1b_0_2000.pth.tar -r gflss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/voc12/FLSS.yaml --imagedataset voc12 --model-path logs/voc12/myexp/checkpoint_20000_2b_0_2000.pth.tar -r flss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/voc12/FLSS.yaml --imagedataset voc12 --model-path logs/voc12/myexp/checkpoint_20000_2b_0_2000.pth.tar -r gflss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/voc12/FLSS.yaml --imagedataset voc12 --model-path logs/voc12/myexp/checkpoint_20000_20b_0_2000.pth.tar -r flss
CUDA_VISIBLE_DEVICES=$1 python eval.py --config config/voc12/FLSS.yaml --imagedataset voc12 --model-path logs/voc12/myexp/checkpoint_20000_20b_0_2000.pth.tar -r gflss
