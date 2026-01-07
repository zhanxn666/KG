import pandas as pd
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline


# Paths to your dataset files (adjust as needed) （路径）
TRAIN_PATH = '/home/e706/zhanxiangning/Tianchi/KnowledgeGraph/OpenBG500/OpenBG500_train.tsv'
DEV_PATH = '/home/e706/zhanxiangning/Tianchi/KnowledgeGraph/OpenBG500/OpenBG500_dev.tsv'
TEST_PATH = '/home/e706/zhanxiangning/Tianchi/KnowledgeGraph/OpenBG500/OpenBG500_test.tsv'
OUTPUT_DIR = 'rotatE_model'
PRED_OUTPUT = 'OpenBG500_pred.tsv'

# Step 1: Load the triples factories （加载数据集）
# Training set （训练集）
training = TriplesFactory.from_path(TRAIN_PATH)

# Validation set (share entity and relation mappings from training) （验证集）
validation = TriplesFactory.from_path(
    DEV_PATH,
    entity_to_id=training.entity_to_id,
    relation_to_id=training.relation_to_id,
)

# Step 2: Train the RotatE model using the pipeline （训练模型）
result = pipeline(
    training=training,
    testing=validation,
    model='RotatE',
    model_kwargs=dict(embedding_dim=256),  # Adjust dimension as needed （调整维度）
    training_kwargs=dict(num_epochs=100, batch_size=128),  # Tune these （调整训练参数）
    optimizer='adam',
    optimizer_kwargs=dict(lr=0.001),
    random_seed=42,  # For reproducibility （可复现性）
)

# Save the trained model （保存模型）
result.save_to_directory(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")

