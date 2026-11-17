---
title: Cleaning tidies email columns but does not validate them
author: Amateur
date: 2026-11-17
branch: main
cost: { minutes: 1, tokens: 2798 }
id: l-81e17b04
scope: repo
status: active
src: agent
touches:
  - src/autoclean/validators.py
---

## Decided
auto_clean has no notion of an email. It trims whitespace, turns blank or placeholder values into missing and merges case variants, which tidies an email column, but malformed addresses ("@example.com", "customer@", "CUSTOMER@@EXAMPLE.COM", "not-an-email") pass through unchanged and uppercase addresses stay uppercase.
The shape agreed for adding it: an opt-in auto_clean step that detects email columns (most values contain "@"), lowercases them and sets invalid addresses to missing, counted in the log; plus a validate_emails validator that returns the offending rows, like validate_numeric_ranges does, so they can be reviewed instead of lost.

## Watch out
Case unification picks the most common spelling of a value and never lowercases by itself, so it is not an email normaliser: an address that only ever appears in uppercase stays uppercase.
Text like "invalid-email" in a numeric column becomes missing only because type inference can't parse it as a number, not because anything recognises email errors.

In the code: database/messy_retail_orders.csv column customer_email is the test case (151 malformed addresses remain after auto_clean); check regex used: ^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$
