"""Gym member segments: k-means on 70% of members, checked on the other 30%.
Run `python analysis.py`. Writes figures/ and outputs/, and prints every result.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "gym_members.csv"
FIG = ROOT / "figures"
OUT = ROOT / "outputs"

SEED = 42                                  # same split and fits on every run
TEST_SIZE = 0.30                           # 30% held out, never used to fit
K_RANGE = range(1, 9)                      # k = 1 (baseline) to 8
K_CHOSEN = 4                               # elbow on both train and test curves
FEATURES = [                               # member_id left out: a label, not behavior
    "age",
    "visits_per_month",
    "avg_visit_minutes",
    "classes_per_month",
    "months_member",
    "monthly_extras_spend",
]

CREDIT_COST = 10                           # $ per member per month
SPEND_CUTOFF = 20                          # at or above this, the credit changes nothing
LOW_SPENDER_NET = 25 * 0.60 - CREDIT_COST  # +$5: $25 more spend at 60% margin, less the $10
HIGH_SPENDER_NET = -CREDIT_COST            # -$10: pure giveaway

NEW_MEMBER = {"age": 30, "visits_per_month": 4, "avg_visit_minutes": 42,
              "classes_per_month": 1, "months_member": 5, "monthly_extras_spend": 10}


def money(x):
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def spread(X, centers):
    """Sum of squared distances from each member to its nearest center."""
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return d2.min(axis=1).sum()


def credit_net(spend):
    """Gym's monthly net from giving one member the $10 credit."""
    return np.where(spend < SPEND_CUTOFF, LOW_SPENDER_NET, HIGH_SPENDER_NET)


def name_segments(centers):
    """Name clusters from their centroids so names don't depend on label order."""
    names = {}
    names[centers["visits_per_month"].idxmin()] = "Drifters"
    names[centers.drop(list(names))["age"].idxmax()] = "Steady Seniors"
    names[centers.drop(list(names))["classes_per_month"].idxmax()] = "Class Spenders"
    for i in centers.index.difference(list(names)):
        names[i] = "Gym Regulars"
    return names


def main():
    FIG.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    df = pd.read_csv(DATA)
    print(f"Rows {len(df)}, columns {df.shape[1]}, mean extras spend ${df.monthly_extras_spend.mean():.2f}")

    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=SEED)
    print(f"Train {len(train)} members, test {len(test)} members")

    scaler = StandardScaler().fit(train[FEATURES])  # scaling fit on train only
    X_train = scaler.transform(train[FEATURES])
    X_test = scaler.transform(test[FEATURES])

    # ---- fit k = 1..8 on train, score on train and test ----
    models, rows = {}, []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X_train)
        models[k] = km
        rows.append({"k": k,
                     "train_spread": km.inertia_,
                     "test_spread": spread(X_test, km.cluster_centers_)})
    curve = pd.DataFrame(rows)
    curve["train_per_member"] = curve.train_spread / len(train)
    curve["test_per_member"] = curve.test_spread / len(test)
    curve.to_csv(OUT / "elbow_spread.csv", index=False)
    print("\nWithin-group spread by k")
    print(curve.round(2).to_string(index=False))

    base_test = curve.loc[curve.k == 1, "test_spread"].item()
    v1_test = curve.loc[curve.k == K_CHOSEN, "test_spread"].item()
    drop = 1 - v1_test / base_test
    print(f"\nTest spread: k=1 {base_test:.1f}, k={K_CHOSEN} {v1_test:.1f} ({drop:.0%} lower)")

    plot_elbow(curve, len(train), len(test))

    # ---- describe the chosen segments ----
    km = models[K_CHOSEN]
    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=FEATURES)
    names = name_segments(centers)

    def assign(features):
        """Segment name for each row of raw features, using the train fit."""
        labels = km.predict(scaler.transform(features[FEATURES]))
        return pd.Series(labels, index=features.index).map(names)

    everyone = df.copy()
    everyone["segment"] = assign(everyone)
    everyone["set"] = np.where(everyone.index.isin(train.index), "train", "test")
    everyone["net_if_credited"] = credit_net(everyone.monthly_extras_spend)

    # Credit a segment when its own members, counted one by one, net the gym money.
    seg_net = everyone.groupby("segment")["net_if_credited"].sum()
    credited = sorted(seg_net[seg_net > 0].index)
    everyone["gets_credit"] = everyone.segment.isin(credited)

    profile = centers.rename(index=names).round(1)
    counts = everyone.groupby("segment").agg(
        members=("member_id", "size"),
        train_members=("set", lambda s: (s == "train").sum()),
        under_20=("monthly_extras_spend", lambda s: (s < SPEND_CUTOFF).sum()),
        median_spend=("monthly_extras_spend", "median"),
        net_if_credited=("net_if_credited", "sum"),
    )
    profile = profile.join(counts)
    profile["gets_credit"] = profile.index.isin(credited)
    profile = profile.sort_values("monthly_extras_spend")
    profile.to_csv(OUT / "segment_profiles.csv")
    print(f"\nSegment centroids (average member), k={K_CHOSEN}")
    print(profile.T.to_string())

    # ---- decision function and the new member ----
    def segment_and_credit(member):
        """Segment for one member's features, and whether that segment gets the credit."""
        seg = assign(pd.DataFrame([member])).iloc[0]
        return seg, seg in credited

    new_seg, new_credit = segment_and_credit(NEW_MEMBER)
    print(f"\nNew member {NEW_MEMBER}")
    print(f"  segment: {new_seg}, gets credit: {new_credit}")

    # ---- value: member by member, all 330 ----
    baseline_net = everyone.net_if_credited.sum()
    v1_net = everyone.loc[everyone.gets_credit, "net_if_credited"].sum()
    n_low = (everyone.monthly_extras_spend < SPEND_CUTOFF).sum()
    n_cred = everyone.gets_credit.sum()
    print(f"\nMembers under $20: {n_low}, at or over $20: {len(everyone) - n_low}")
    print(f"Baseline (credit to all {len(everyone)}): {money(baseline_net)}/month")
    print(f"v1 (credit to {', '.join(credited)}; {n_cred} members): {money(v1_net)}/month")
    print(f"Difference: {money(v1_net - baseline_net)}/month, {money(12 * (v1_net - baseline_net))}/year")
    print(f"Credit cost: baseline ${CREDIT_COST * len(everyone):,}, v1 ${CREDIT_COST * n_cred:,}")

    plot_segments(profile)
    everyone.to_csv(OUT / "member_assignments.csv", index=False)

    summary = {
        "rows": len(df), "train": len(train), "test": len(test),
        "k": K_CHOSEN, "test_spread_k1": round(base_test, 1),
        "test_spread_k": round(v1_test, 1), "test_spread_drop": round(drop, 3),
        "credited_segments": credited,
        "new_member": {"segment": new_seg, "gets_credit": bool(new_credit)},
        "baseline_net": float(baseline_net), "v1_net": float(v1_net),
        "monthly_gain": float(v1_net - baseline_net),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


def plot_elbow(curve, n_train, n_test):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    ax.plot(curve.k, curve.train_spread, "o-", color="#1f4e79", label=f"Training members (n={n_train})")
    ax.plot(curve.k, curve.test_spread, "s--", color="#c55a11", label=f"Test members (n={n_test})")
    ax.axvline(K_CHOSEN, color="grey", lw=0.8, ls=":")
    for col in ("train_spread", "test_spread"):
        for k in (1, K_CHOSEN):
            y = curve.loc[curve.k == k, col].item()
            ax.annotate(f"{y:,.0f}", (k, y), textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.set_ylim(0, curve.train_spread.max() * 1.1)
    ax.set_xticks(list(K_RANGE))
    ax.set_xlabel("Number of segments (k)")
    ax.set_ylabel("Within-group spread\n(sum of squared distances)")
    ax.set_title(f"Spread drops sharply up to k = {K_CHOSEN}, then flattens", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "elbow_train_test.png")
    plt.close(fig)


def plot_segments(profile):
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=200)
    colors = ["#2e7d32" if c else "#9e9e9e" for c in profile.gets_credit]
    bars = ax.bar(profile.index, profile.monthly_extras_spend, color=colors)
    ax.axhline(SPEND_CUTOFF, color="black", lw=0.8, ls="--")
    ax.text(len(profile) - 0.5, SPEND_CUTOFF + 1, "$20 cutoff", ha="right", fontsize=8)
    for b, v, net in zip(bars, profile.monthly_extras_spend, profile.net_if_credited):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"${v:.0f}", ha="center", fontsize=9)
        ax.text(b.get_x() + b.get_width() / 2, -6, f"credit nets {'+' if net > 0 else '-'}${abs(net):,.0f}/mo",
                ha="center", fontsize=8)
    ax.set_ylim(-9, profile.monthly_extras_spend.max() * 1.12)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Average extras spend ($/month)")
    ax.set_title("Green segments get the $10 credit", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "segment_spend.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
