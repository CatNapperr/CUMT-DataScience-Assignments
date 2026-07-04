import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from data_preprocessing import load_processed_data, LABEL_NAMES
from models import LSTMAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "outputs"


def extract_embeddings(model, loader):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            out, _ = model.lstm(x)               # (N, 3, hidden)
            emb = out[:, -1, :]                  
            embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def build_player_embeddings(embeddings, players, labels):
    player_emb = {}
    for emb, name, lbl in zip(embeddings, players, labels):
        if name not in player_emb:
            player_emb[name] = {"embs": [], "labels": []}
        player_emb[name]["embs"].append(emb)
        player_emb[name]["labels"].append(lbl)

    names = []
    emb_matrix = []
    labels_out = []
    for name, data in player_emb.items():
        names.append(name)
        emb_matrix.append(np.mean(data["embs"], axis=0))
        labels_out.append(int(np.bincount(data["labels"]).argmax()))  # majority label

    emb_matrix = np.stack(emb_matrix)
    emb_matrix = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8)
    return emb_matrix, np.array(names), np.array(labels_out)


def find_similar_players(query, names, emb_matrix, labels, top_k=5):
    idx = np.where(names == query)[0]
    if len(idx) == 0:
        print(f"Player '{query}' not found in dataset.")
        return []
    query_emb = emb_matrix[idx[0]]
    sims = emb_matrix @ query_emb 

    sims[idx[0]] = -2

    top_indices = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in top_indices:
        results.append({
            "player": str(names[i]),
            "similarity": round(float(sims[i]), 4),
            "label": int(labels[i]),
            "label_name": LABEL_NAMES[int(labels[i])],
        })
    return results


def main():
    (_, _), (_, _), (X_test, y_test), _, players_test, pos_test = load_processed_data()

    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    test_loader = DataLoader(test_ds, batch_size=64)


    model = LSTMAttention().to(DEVICE)
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/LSTM_Attention.pth", map_location=DEVICE))


    embeddings = extract_embeddings(model, test_loader)
    print(f"Extracted {embeddings.shape[0]} sample embeddings")

    emb_matrix, names, labels = build_player_embeddings(embeddings, players_test, y_test)
    print(f"Built {len(names)} player embeddings (dim={emb_matrix.shape[1]})")


    np.savez(f"{OUTPUT_DIR}/player_embeddings.npz",
             embeddings=emb_matrix, names=names, labels=labels)
    print(f"Saved embeddings to {OUTPUT_DIR}/player_embeddings.npz\n")


    queries = ["Stephen Curry", "LeBron James", "Kevin Durant",
               "James Harden", "Russell Westbrook", "Kawhi Leonard",
               "Dirk Nowitzki", "Dwyane Wade"]
    for q in queries:
        results = find_similar_players(q, names, emb_matrix, labels, top_k=5)
        if results:
            print(f"Most similar to {q} ({LABEL_NAMES[labels[names == q][0]]}):")
            for r in results:
                print(f"  {r['player']:<25} sim={r['similarity']:.4f}  [{r['label_name']}]")
            print()


if __name__ == "__main__":
    main()
