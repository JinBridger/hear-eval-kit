# 运行方法:
# 1. 运行 generate embeddings
# 2. 运行 predict
# 3. 运行 get_scores.py 获取最终结果

# HEAR 的 task 文件夹里面只放需要测试的 task 即可
# 默认是测试全部任务

# =========================================
# sdvae
# =========================================
# 创建环境
conda create -n sdvae-audio-hear-benchmark python=3.9
conda activate sdvae-audio-hear-benchmark
pip install -r sdvae_requirements.txt --no-deps
# 生成 embeddings
python3 -m heareval.embeddings.runner sdvae_hear_api \
        --model sdvae_hear_api.py \
        --tasks-dir /data/jq_data/HEAR/hear-2021.0.6/tasks/ \
        --embeddings-dir embeddings
# 预测
python3 -m heareval.predictions.runner embeddings/sdvae_hear_api/* --gpus "[0]"

# =========================================
# ezaudio
# =========================================
# 创建环境
conda create -n ezaudio-hear-benchmark python=3.8
conda activate ezaudio-hear-benchmark
pip install -r ezaudio_requirements.txt --no-deps
# 生成 embeddings
python3 -m heareval.embeddings.runner ezaudio_hear_api \
        --model ezaudio_hear_api.py \
        --tasks-dir /data/jq_data/HEAR/hear-2021.0.6/tasks/ \
        --embeddings-dir embeddings
# 预测
python3 -m heareval.predictions.runner embeddings/ezaudio_hear_api/* --gpus "[0]"

# =========================================
# audioldm2
# =========================================
# 创建环境
conda create -n audioldm2-hear-benchmark python=3.8
conda activate audioldm2-hear-benchmark
pip install -r audioldm2_requirements.txt --no-deps
# 生成 embeddings
python3 -m heareval.embeddings.runner audioldm2_hear_api \
        --model audioldm2_hear_api.py \
        --tasks-dir /data/jq_data/HEAR/hear-2021.0.6/tasks/ \
        --embeddings-dir embeddings/audioldm2
# 预测
python3 -m heareval.predictions.runner embeddings/audioldm2_hear_api/* --gpus "[0]"