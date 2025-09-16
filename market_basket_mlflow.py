import os
import csv
import hashlib
import datetime
import mlflow
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
MLFLOW_DB_URI = "sqlite:///mlflow.db"   # stored in project root
TRANSACTIONS_FILE = "transactions.csv"  # input
OUTPUT_DIR = "outputs"
ARTIFACT_LOCATION = os.path.join(os.getcwd(), "mlruns")  # ✅ keep runs local
# ----------------------------

# set MLflow tracking
mlflow.set_tracking_uri(MLFLOW_DB_URI)
mlflow.set_experiment("MarketBasketAnalysis")

# hyperparameter sweep
MIN_SUPPORTS = [0.1, 0.15, 0.2]
MIN_LIFTS = [1.0, 1.5]
TOP_K_RULES_TO_PLOT = 20

# ensure outputs dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_file_sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_transactions(path):
    transactions = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            items = [item.strip() for item in row if item and item.strip()]
            if items:
                transactions.append(items)
    return transactions

# --- load dataset once ---
if not os.path.exists(TRANSACTIONS_FILE):
    raise FileNotFoundError(f"Missing {TRANSACTIONS_FILE} in {os.getcwd()}")

transactions = load_transactions(TRANSACTIONS_FILE)
num_transactions = len(transactions)
data_sha1 = compute_file_sha1(TRANSACTIONS_FILE)

# --- encode once ---
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
onehot = pd.DataFrame(te_ary, columns=te.columns_)
item_frequencies = onehot.sum().sort_values(ascending=False)

# --- grid search ---
for support in MIN_SUPPORTS:
    for lift in MIN_LIFTS:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        RUN_PREFIX = "FINAL"
        run_name = f"{RUN_PREFIX}_{num_transactions}tx_s{support}_l{lift}_{timestamp}"


        # outputs
        rules_csv = os.path.join(OUTPUT_DIR, "rules.csv")
        fi_csv = os.path.join(OUTPUT_DIR, "frequent_itemsets.csv")
        net_png = os.path.join(OUTPUT_DIR, "rules_network.png")
        top_items_csv = os.path.join(OUTPUT_DIR, "top_items.csv")
        hist_png = os.path.join(OUTPUT_DIR, "support_hist.png")

        # compute itemsets and rules
        frequent_itemsets = apriori(onehot, min_support=support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=lift)

        # metrics
        num_itemsets = len(frequent_itemsets)
        num_rules = len(rules)
        avg_support = float(frequent_itemsets['support'].mean()) if num_itemsets > 0 else 0.0
        max_lift = float(rules['lift'].max()) if num_rules > 0 else 0.0
        avg_confidence = float(rules['confidence'].mean()) if num_rules > 0 else 0.0
        longest_rule_len = (
            int((rules['antecedents'].apply(len) + rules['consequents'].apply(len)).max())
            if num_rules > 0 else 0
        )

        # save outputs
        frequent_itemsets.to_csv(fi_csv, index=False)
        rules.to_csv(rules_csv, index=False)

        top_items = item_frequencies.reset_index()
        top_items.columns = ['item', 'count']
        top_items.to_csv(top_items_csv, index=False)

        # network plot
        if num_rules > 0:
            top = rules.sort_values(by=['lift', 'confidence'], ascending=False).head(TOP_K_RULES_TO_PLOT)
            G = nx.DiGraph()
            for _, row in top.iterrows():
                ant = ",".join(sorted(list(row["antecedents"])))
                cons = ",".join(sorted(list(row["consequents"])))
                G.add_edge(ant, cons, weight=float(row["lift"]))
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G, k=0.7, seed=42)
            nx.draw(G, pos, with_labels=True, node_size=900, font_size=8)
            plt.title(f"Top rules s={support} l={lift}")
            plt.savefig(net_png, bbox_inches="tight")
            plt.close()
        else:
            plt.figure(figsize=(4, 2))
            plt.text(0.5, 0.5, "No rules", ha='center', va='center')
            plt.axis('off')
            plt.savefig(net_png, bbox_inches="tight")
            plt.close()

        # histogram
        if num_itemsets > 0:
            plt.figure(figsize=(6, 4))
            plt.hist(frequent_itemsets['support'], bins=10, edgecolor='k')
            plt.title(f"Support distribution (min_sup={support})")
            plt.xlabel("Support")
            plt.ylabel("Count")
            plt.savefig(hist_png, bbox_inches="tight")
            plt.close()

        # --- MLflow logging ---
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_param("dataset", TRANSACTIONS_FILE)
            mlflow.log_param("dataset_sha1", data_sha1)
            mlflow.log_param("num_transactions", num_transactions)
            mlflow.log_param("min_support", support)
            mlflow.log_param("min_lift", lift)

            mlflow.set_tag("top_k_rules_plotted", TOP_K_RULES_TO_PLOT)

            mlflow.log_metric("num_frequent_itemsets", num_itemsets)
            mlflow.log_metric("num_rules", num_rules)
            mlflow.log_metric("avg_support_of_itemsets", avg_support)
            mlflow.log_metric("max_lift", max_lift)
            mlflow.log_metric("avg_confidence", avg_confidence)
            mlflow.log_metric("longest_rule_len", longest_rule_len)

            mlflow.log_artifact(fi_csv)
            mlflow.log_artifact(rules_csv)
            mlflow.log_artifact(net_png)
            mlflow.log_artifact(top_items_csv)
            if os.path.exists(hist_png):
                mlflow.log_artifact(hist_png)

        print(f"Finished run: {run_name} (rules={num_rules}, itemsets={num_itemsets})")
