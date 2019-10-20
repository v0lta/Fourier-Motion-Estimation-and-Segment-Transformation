#block(name=msproj, threads=2, memory=7500, subtasks=1, gpus=1, hours=72)
echo $CUDA_VISIBLE_DEVICES
#source activate msproj
python "$@"
