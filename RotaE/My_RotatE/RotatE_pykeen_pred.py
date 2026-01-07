import pandas as pd
from pykeen.triples import TriplesFactory
from pykeen.predict import predict_target
import torch
import os

# --------------------------- Paths ---------------------------
TRAIN_PATH = '/home/e706/zhanxiangning/Tianchi/KnowledgeGraph/OpenBG500/OpenBG500_train.tsv'
TEST_PATH = '/home/e706/zhanxiangning/Tianchi/KnowledgeGraph/OpenBG500/OpenBG500_test.tsv'
MODEL_DIR = 'rotatE_model'                    # Folder containing trained_model.pkl
PRED_OUTPUT = 'OpenBG500_pred.tsv'

# ------------------- Load training triples factory -------------------
print("Loading training triples...")
training_factory = TriplesFactory.from_path(TRAIN_PATH)

# ------------------- Load the trained model -------------------
model_path = os.path.join(MODEL_DIR, 'trained_model.pkl')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

print(f"Loading model from {model_path}...")
model = torch.load(model_path, map_location='cpu')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
print(f"Model loaded on {device}")

# ------------------- Load test queries -------------------
print("Loading test queries...")
test_df = pd.read_csv(TEST_PATH, sep='\t', header=None, names=['head', 'relation'])

# ------------------- Generate predictions -------------------
print("Generating predictions...")
predictions = []

for idx, row in test_df.iterrows():
    head = row['head']
    relation = row['relation']

    # Predict tails
    pack = predict_target(
        model=model,
        head=head,
        relation=relation,
        triples_factory=training_factory,
    )

    # Filter out known triples from training set
    filtered_pack = pack.filter_triples(training_factory)

    # Get top-10 predictions
    top_df = filtered_pack.df.nlargest(10, 'score')

    # Extract tail labels (entity IDs like ent_012552)
    top_tails = top_df['tail_label'].tolist()

    # Always pad to exactly 10 with 'ent_xxxxxx'
    while len(top_tails) < 10:
        top_tails.append('ent_xxxxxx')

    # Keep only first 10 (safety)
    top_tails = top_tails[:10]

    # Build the line: head \t relation \t t1 \t t2 \t ... \t t10
    line = [head, relation] + top_tails
    predictions.append(line)

    if (idx + 1) % 1000 == 0:
        print(f"Processed {idx + 1}/{len(test_df)} queries")

# ------------------- Save result -------------------
pred_df = pd.DataFrame(predictions)
pred_df.to_csv(PRED_OUTPUT, sep='\t', header=False, index=False)
print(f"\nPredictions saved to {PRED_OUTPUT}")
print(f"Total lines: {len(predictions)}")
print("Sample output:")
print(pred_df.head(5).to_string(header=False, index=False))