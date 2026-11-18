---
title: "Email columns are detected from their values, lower-cased, and malformed addresses become missing"
author: Amateur
date: 2026-11-18
branch: audit-universal-cleaning
cost: { minutes: 16, tokens: 271001 }
id: l-c353b808
scope: repo
status: active
replaces: l-81e17b04
src: agent
touches:
  - src/autoclean/validators.py
---

## Decided
auto_clean has an email step: a column counts as emails when at least 80% of its values contain "@" and at least half are valid addresses (so a column of "@handles" doesn't qualify). Addresses are trimmed, lower-cased, and stripped of "mailto:", display names ("Ann <ann@x.com>"), zero-width characters and trailing punctuation.
Malformed addresses become missing by default, the same way unparseable numbers and dates do, and the log counts them. emails="keep" leaves them in place and validate_emails lists their rows for review, so nobody has to lose data to see the problems.
The valid-address check is a practical subset of RFC 5322: quoted local parts and IP-literal domains are rejected because in real exports they are almost always errors.

## Tried and ruled out
Requiring 90% of values to contain "@" missed small columns where a single "not an email" entry pushed the share below the cut-off; 80% plus the majority-valid check is enough to keep @handles out.
Stripping "." from both ends of an address made ".user@example.com" look valid; only trailing punctuation from prose is trimmed.

## Watch out
A single-column frame loses the row when its only value is an invalid email, because the row is then empty and drop_empty removes it.

In the code: src/autoclean/emails.py (_AT_SIGN_SHARE, _VALID_SHARE, _VALID_EMAIL); auto_clean(emails="missing"|"keep"|None); validate_emails in validators.py; fixture tests/data/crm_contacts.csv.
