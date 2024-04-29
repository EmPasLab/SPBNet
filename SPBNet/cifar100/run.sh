#!/bin/bash 

batch_size=512
label_smooth=0.0
python=train.py
workers=20


############################################################################
############################################################################
# dorefanet.py weight scaling
day=DEC31
ver=tako_xresnet
seed_array=(
42
43
44
45
46
)

echo "batch_size: ${batch_size}"
echo "day: ${day}"
echo "label_smooth: ${label_smooth}"
echo "ver: ${ver}"
echo "python: ${python}"
echo "workers: ${workers}"

data=$(date +%Y)-$(date +%m)-$(date +%d)--$(date +%H)-$(date +%M)-$(date +%S)
mkdir ./results/$data
cp ./run.sh ./*.py ./results/$data


for seed in ${seed_array[*]}
do 

  epochs=400
  
  iter1=( 1 1  0.001 ${epochs} 0.00001 lambda adam )

  mode_array=( 
  iter1
  )

  #}}}
  # End of July22super1
############################################################################
############################################################################

  #num_iter=0
  for arr in ${mode_array[*]}
  do
    row="$arr[*]"
    mode=(${!row})
    #echo ${mode[0]}
    echo "##Start########################################################Start##"
    #echo "num_iter: $num_iter"
    ###################################
    # ${normal_mode}: Top-1 accuracy 
    echo "######################################################################"
    #if [ ${num_iter} -eq "" ];
    rm -rf ./results/$data/checkpoint_${mode[0]}_${mode[1]}_${day}_2022${ver}_seeds${seed}.pth.tar
    rm -rf ./results/$data/model_best_${mode[0]}_${mode[1]}_${day}_2022${ver}_seeds${seed}.pth.tar
    if [ ${#mode[@]} -eq 7 ];
    then
      echo "${day}, 2022: --a_bits=${mode[0]}, --w_bits=${mode[1]}, --learning_rate=${mode[2]}, --epochs=${mode[3]}, --weight_decay=${mode[4]}" 
      echo " --scheduler=${mode[5]}, --optimizer=${mode[6]}"
      echo " --data=./data, --workers=${workers}, --batch_size=${batch_size}, --seed=${seed}, --momentum=0.9, --label_smooth=${label_smooth}" 
      python3 ${python} --a_bits=${mode[0]} --w_bits=${mode[1]} --learning_rate=${mode[2]} --epochs=${mode[3]} --weight_decay=${mode[4]} \
                        --scheduler=${mode[5]} --optimizer=${mode[6]} \
                        --label_smooth=${label_smooth} --data=./data/cifar100 --workers=${workers} --batch_size=${batch_size} --seed=${seed} --momentum=0.9 \
                        --outputfile=./results/$data/${mode[0]}_${mode[1]}_training_${day}_2022${ver}_${arr}_seeds${seed}.out --save=./results/$data \
                       | tee -a ./results/$data/${mode[0]}_${mode[1]}_training_${day}_2022${ver}_${arr}_seeds${seed}.txt
      echo "${day}, 2022: --a_bits=${mode[0]}, --w_bits=${mode[1]}, --learning_rate=${mode[2]}, --epochs=${mode[3]}, --weight_decay=${mode[4]}" 
      echo " --scheduler=${mode[5]}, --optimizer=${mode[6]}"
      echo " --data=./data, --workers=${workers}, --batch_size=${batch_size}, --seed=${seed}, --momentum=0.9, --label_smooth=${label_smooth}" 
    else
      echo "${day}, 2022: --a_bits=${mode[0]}, --w_bits=${mode[1]}, --learning_rate=${mode[2]}, --epochs=${mode[3]}, --weight_decay=${mode[4]}" 
      echo " --scheduler=${mode[5]}, --optimizer=${mode[6]}, --pretrained=${mode[7]}, "
      echo " --data=./data, --workers=${workers}, --batch_size=${batch_size}, --seed=${seed}, --momentum=0.9, --label_smooth=${label_smooth}" 
      python3 ${python} --a_bits=${mode[0]} --w_bits=${mode[1]} --learning_rate=${mode[2]} --epochs=${mode[3]} --weight_decay=${mode[4]} \
                        --scheduler=${mode[5]} --optimizer=${mode[6]} --pretrained=${mode[7]} \
                        --label_smooth=${label_smooth} --data=./data/cifar100 --workers=${workers} --batch_size=${batch_size} --seed=${seed} --momentum=0.9 \
                        --outputfile=./results/$data/${mode[0]}_${mode[1]}_training_${day}_2022${ver}_${arr}_seeds${seed}.out --save=./results/$data \
                       | tee -a ./results/$data/${mode[0]}_${mode[1]}_training_${day}_2022${ver}_${arr}_seeds${seed}.txt
      echo "${day}, 2022: --a_bits=${mode[0]}, --w_bits=${mode[1]}, --learning_rate=${mode[2]}, --epochs=${mode[3]}, --weight_decay=${mode[4]}" 
      echo " --scheduler=${mode[5]}, --optimizer=${mode[6]}, --pretrained=${mode[7]}, "
      echo " --data=./data, --workers=${workers}, --batch_size=${batch_size}, --seed=${seed}, --momentum=0.9, --label_smooth=${label_smooth}" 
    fi
    #let "num_iter=${num_iter}+1" 
    mv  ./results/$data/checkpoint.pth.tar ./results/$data/checkpoint_${mode[0]}_${mode[1]}_${day}_2022${ver}_seeds${seed}.pth.tar
    mv  ./results/$data/model_best.pth.tar ./results/$data/model_best_${mode[0]}_${mode[1]}_${day}_2022${ver}_seeds${seed}.pth.tar
    echo "######################################################################"
    echo "##End############################################################End##"
    echo " "
    echo " "
    ###################################
  done
done
