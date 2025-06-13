
# train GNR
python train_GNR.py --train_steps 50000 --r 180 --s_min 1.0 --s_max 1.2 --fp 0.35 --neg_p 0.5 --model_nf 128 --batch_size 32 --num_workers 16 -ed 256bits --w_info_path w1_256.pth

# generate watermarked images
python gaussmarker_gen.py --chacha --num 10 --output_path './gen_10'  --w1_path w1_256.pth

# extract watermark
python gaussmarker_det.py --chacha --model_nf 128 --GNR_path './GNR_bits256/model_final.pth' --input_path './gen_10'  --fpr 0.01 --num 10  --output_path './results/Clean' --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k --w1_path './w1_256.pth'

# extract watermark with rotation 75
python gaussmarker_det.py --chacha --model_nf 128 --GNR_path './GNR_bits256/model_final.pth' --input_path './gen_10'  --fpr 0.01 --num 10  --output_path './results/Rotate75' --affine 1 --affine_params 75,0,1,0 --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k --w1_path './w1_256.pth'