SEQUENCES=("lan_images620_1300" "marc_images35000_36200" "olek_images0812" "vlad_images1011")
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="data/monocap/$SEQUENCE"
    python train_cl.py -s $dataset --eval --exp_name monocap/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 2000
done
