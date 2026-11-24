# Gym member segments

K-means clustering of 330 gym members to decide who should get the monthly $10 extras credit.

## Run it

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python analysis.py
```

`analysis.py` prints every result and writes the charts to `figures/` and the tables to `outputs/`. The split and the fits use seed 42, so the numbers are the same on every run.

## Method

- Data: `data/gym_members.csv`, 330 members, averages over the last 3 months.
- Split: 70% train (231), 30% test (99). The scaler and k-means are fit on train only.
- Features: age, visits per month, minutes per visit, classes per month, months as a member, extras spend. `member_id` is left out because it is a label, not behavior.
- Model: k-means for k = 1 to 8, with standardized features. k = 1 is the baseline.
- Test check: each test member goes to the nearest train center, and the squared distances are summed for each k.

## Results

Both the train and test curves bend at k = 4. Test spread is 601 at k = 1 and 150 at k = 4, 75% lower.

| Segment | Members | Average member | Avg extras | Credit |
|---|---|---|---|---|
| Drifters | 70 | late 20s, 3 visits/mo, about 6 months in | $9 | yes |
| Steady Seniors | 60 | mid 60s, about 5 years in, 12 visits/mo | $17 | no |
| Gym Regulars | 109 | early 30s, 16 visits/mo, 65-minute visits | $25 | no |
| Class Spenders | 91 | early 40s, 7 classes/mo | $47 | no |

A segment gets the credit when its own members, counted one by one, earn the gym money: +$5 for each member spending under $20 and -$10 for each member spending $20 or more. Only Drifters pass. Steady Seniors come close at -$45 a month, because 23 of the 60 already spend $20 or more.

New member (age 30, 4 visits, 42 minutes, 1 class, 5 months, $10 extras): Drifters, gets the credit.

Monthly net from the credit across all 330 members:

- Baseline, credit to everyone: -$1,200
- v1, credit to Drifters only: +$185
- Difference: +$1,385 a month, about $16,620 a year

## Files

- `analysis.py`: split, scaling, k-means, test check, segments, decision function, value
- `figures/elbow_train_test.png`: train and test spread against k
- `figures/segment_spend.png`: extras spend by segment, with the $20 cutoff
- `outputs/elbow_spread.csv`, `segment_profiles.csv`, `member_assignments.csv`, `summary.json`

AI use: an AI coding assistant helped write the code, including the chart code. I ran every script myself, so the charts and numbers come from running this code.
