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

# ---------- EDIT ONLY IF YOUR PATH CHANGES ----------
MLFLOW_DB_URI = "sqlite:///C:/Users/nagen/OneDrive/Desktop/Kaggle_datasets/Market_basket_analysis/market-basket-mlflow/mlflow.db"
PROJECT_DIR = "C:/Users/nagen/OneDrive/Desktop/Kaggle_datasets/Market_basket_analysis/market-basket-mlflow"
# ----------------------------------------------------

# set MLflow
mlflow.set_tracking_uri(MLFLOW_DB_URI)
mlflow.set_experiment("MarketBasketAnalysis")

# -------- CONFIG (tweak these lists to run different experiments) --------
TRANSACTIONS_FILE = "transactions.csv"
MIN_SUPPORTS = [0.1, 0.15, 0.2]   # list of supports to sweep
MIN_LIFTS = [1.0, 1.5]           # list of lift thresholds to sweep
TOP_K_RULES_TO_PLOT = 20
# -----------------------------------------------------------------------

os.chdir(PROJECT_DIR)
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/top_items", exist_ok=True)

def compute_file_sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
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
    raise FileNotFoundError(f"Create {TRANSACTIONS_FILE} in {os.getcwd()} (one transaction per line, comma-separated)")

transactions = load_transactions(TRANSACTIONS_FILE)
num_transactions = len(transactions)
data_sha1 = compute_file_sha1(TRANSACTIONS_FILE)

# --- encode once (reused across hyperparams) ---
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
onehot = pd.DataFrame(te_ary, columns=te.columns_)
item_frequencies = onehot.sum().sort_values(ascending=False)  # series of item -> count

# --- helper to make safe small strings for filenames ---
def safe_str(x):
    return str(x).replace('.', 'p').replace(' ', '_')

# --- grid search over hyperparameters ---
for support in MIN_SUPPORTS:
    for lift in MIN_LIFTS:
        support_str = safe_str(f"{support:.3f}")
        lift_str = safe_str(f"{lift:.2f}")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 👇 Custom prefix for runs (change this manually each experiment)
        custom_prefix = "exp1"
        run_name = f"{custom_prefix}_{num_transactions}tx_s{support_str}_l{lift_str}_{timestamp}"

        # filenames per-run (avoid overwriting)
        rules_csv = f"outputs/rules_s{support_str}_l{lift_str}_{timestamp}.csv"
        fi_csv = f"outputs/frequent_itemsets_s{support_str}_l{lift_str}_{timestamp}.csv"
        net_png = f"outputs/rules_network_s{support_str}_l{lift_str}_{timestamp}.png"
        top_items_csv = f"outputs/top_items/top_items_s{support_str}_l{lift_str}_{timestamp}.csv"
        hist_png = f"outputs/support_hist_s{support_str}_l{lift_str}_{timestamp}.png"

        # --- compute itemsets and rules ---
        frequent_itemsets = apriori(onehot, min_support=support, use_colnames=True)

        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=lift)
        else:
            rules = pd.DataFrame()  # no rules

        # --- compute additional metrics ---
        num_itemsets = len(frequent_itemsets)
        num_rules = len(rules)
        avg_support = float(frequent_itemsets['support'].mean()) if num_itemsets > 0 else 0.0
        max_lift = float(rules['lift'].max()) if num_rules > 0 else 0.0
        avg_confidence = float(rules['confidence'].mean()) if num_rules > 0 else 0.0
        longest_rule_len = 0
        if num_rules > 0:
            longest_rule_len = int((rules['antecedents'].apply(len) + rules['consequents'].apply(len)).max())

        # --- save artifacts locally ---
        frequent_itemsets.to_csv(fi_csv, index=False)
        rules.to_csv(rules_csv, index=False)

        # top items as csv
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

        # support histogram
        if num_itemsets > 0:
            plt.figure(figsize=(6, 4))
            plt.hist(frequent_itemsets['support'], bins=[0,0.2,0.4,0.6,0.8,1.0], edgecolor='black')
            plt.xlabel("Support value")
            plt.ylabel("Number of itemsets")
            plt.title(f"Support histogram (min_sup={support}, min_lift={lift})")
            plt.savefig(hist_png, bbox_inches="tight")
            plt.close()
        else:
            plt.figure(figsize=(4, 2))
            plt.text(0.5, 0.5, "No itemsets", ha='center', va='center')
            plt.axis("off")
            plt.savefig(hist_png, bbox_inches="tight")
            plt.close()

        # --- Log to MLflow ---
        with mlflow.start_run(run_name=run_name):
            # params
            mlflow.log_param("dataset", TRANSACTIONS_FILE)
            mlflow.log_param("dataset_sha1", data_sha1)
            mlflow.log_param("num_transactions", num_transactions)
            mlflow.log_param("min_support", support)
            mlflow.log_param("min_lift", lift)

            # tags
            mlflow.set_tag("sweep", "support_lift_grid")
            mlflow.set_tag("top_k_rules_plotted", TOP_K_RULES_TO_PLOT)

            # metrics
            mlflow.log_metric("num_frequent_itemsets", num_itemsets)
            mlflow.log_metric("num_rules", num_rules)
            mlflow.log_metric("avg_support_of_itemsets", avg_support)
            mlflow.log_metric("max_lift", max_lift)
            mlflow.log_metric("avg_confidence", avg_confidence)
            mlflow.log_metric("longest_rule_len", longest_rule_len)

            # artifacts
            mlflow.log_artifact(fi_csv)
            mlflow.log_artifact(rules_csv)
            mlflow.log_artifact(net_png)
            mlflow.log_artifact(top_items_csv)
            mlflow.log_artifact(hist_png)

        print(f"Finished run: {run_name}  (rules={num_rules}, itemsets={num_itemsets})")
