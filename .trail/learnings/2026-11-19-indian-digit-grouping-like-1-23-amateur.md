---
title: "Indian digit grouping like 1,23,456.00 isn't parsed, and day-first dates swap silently without dayfirst=True"
author: Amateur
date: 2026-11-19
branch: main
cost: { minutes: 16, tokens: 208070 }
id: l-eefbd630
scope: repo
status: active
src: agent
touches:
  - scripts/stress_test.py
  - src/autoclean/inference.py
---

## Watch out
- Values in Indian lakh grouping ("1,14,357.90") don't parse as numbers. In the 100k stress test all 469 such Order Total cells came back missing. US grouping, "₹", "INR", "Rs." and European "1.234,56" all parse. Only a 0.5% mix in one column was tested, not a whole lakh-formatted column.
- Without dayfirst=True, DD/MM dates with a day of 12 or less swap day and month with no warning; the log only shows the usual counts. That was 10,491 of 98,194 dates (10.7%) in the stress test.
- The stress-test file repeats its header row mid-file, which makes pandas read every column as text. So it can't show load-time type problems such as a column of "0042" codes loading as 42.0.

In the code: scripts/stress_test.py (`python scripts/stress_test.py [--rows N] [--no-dayfirst]`) writes database/messy_orders_100k.csv and scores auto_clean against the known answers. Number parsing: _as_number in src/autoclean/inference.py.
