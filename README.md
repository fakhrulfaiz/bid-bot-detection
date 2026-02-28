# Bid Bot Detection — MLOps Project

An end-to-end MLOps pipeline to detect fraudulent (bot) bidding behavior in online auctions. Built as a hands-on MLOps learning project using the [Facebook Recruiting IV: Human or Robot?](https://www.kaggle.com/c/facebook-recruiting-iv-human-or-robot) dataset from Kaggle.

## Problem Statement

Online auction platforms are vulnerable to automated bots that place fraudulent bids to inflate prices or manipulate outcomes. This project builds a binary classifier to predict whether a bidder is a **human (0)** or a **bot (1)**, using behavioral and account-level signals.

## Dataset

Source: [Kaggle — Facebook Recruiting IV](https://www.kaggle.com/c/facebook-recruiting-iv-human-or-robot)

| File | Description |
|------|-------------|
| `train.csv` | Bidder-level training data with labels |
| `test.csv` | Bidder-level test data |

### Features

| Field | Description |
|-------|-------------|
| `bidder_id` | Unique identifier of a bidder |
| `payment_account` | Payment account associated with a bidder (obfuscated) |
| `address` | Mailing address of a bidder (obfuscated) |
| `outcome` | Target label — `1.0` = bot, `0.0` = human |

> The outcome was partially hand-labeled and partially stats-based. Bots include both banned accounts (clear proof) and suspicious bidders whose activity exceeds platform-wide averages.
