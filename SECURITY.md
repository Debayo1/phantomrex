# Security & Credential Rotation Protocol

## ⚠️ Important Security Notice
Prior commits in this repository contained `telegram_config.json` with plaintext credentials. Even though the repository is private, credentials should be rotated immediately to eliminate all historical exposure.

---

## Owner-Only Manual Rotation Steps (Strict Order per Addendum A7)

Follow these steps in the exact order below. **Credential rotation must precede the history purge** because anyone with a local clone of old commits still has active credentials until rotated.

### 1. Confirm Repository Privacy
1. Go to your GitHub repository: `https://github.com/Debayo1/phantomrex`
2. Navigate to **Settings** > **Danger Zone**.
3. Verify that the repository is set to **Private**.

### 2. Revoke & Rotate Telegram Bot Token
1. Open Telegram and search for [@BotFather](https://t.me/BotFather).
2. Send `/revoke` and select your bot (`PhantomRex`).
3. BotFather will immediately revoke the old token and generate a brand new token.
4. Update your local uncommitted `telegram_config.json` or `.env` file with the new token.

### 3. Rotate MT5 Account Passwords
1. Log in to your [Exness Personal Area](https://my.exness.com/).
2. Under **My Accounts**:
   * Locate your **Demo** account -> Click settings (three dots) -> **Change trading password**.
   * Locate your **Real** account -> Click settings (three dots) -> **Change trading password**.
3. Set strong new passwords and update your local uncommitted `telegram_config.json` or `.env` file.

### 4. Purge Historical Commits from Git
Once tokens and passwords have been rotated, scrub the historical commit blobs so the old values no longer exist anywhere in git:

```bash
# Install git-filter-repo if not installed
python -m pip install git-filter-repo

# Scrub telegram_config.json from all past commits
git filter-repo --path telegram_config.json --invert-paths

# Re-link your remote repository (filter-repo removes the remote for safety)
git remote add origin https://github.com/Debayo1/phantomrex.git

# Force-push the scrubbed history to GitHub
git push origin --force --all
```

---

## Safe Environment Variable Configuration
Instead of keeping credentials in `telegram_config.json`, you can optionally place them in a `.env` file or export them as environment variables:

```bash
# Telegram & AI
TELEGRAM_BOT_TOKEN="your_new_token_here"
GEMINI_API_KEY="your_gemini_key_here"

# MT5 Demo Account
MT5_DEMO_LOGIN="477157874"
MT5_DEMO_PASSWORD="your_new_demo_password"
MT5_DEMO_SERVER="Exness-MT5Trial9"

# MT5 Real Account
MT5_REAL_LOGIN="295802448"
MT5_REAL_PASSWORD="your_new_real_password"
MT5_REAL_SERVER="Exness-MT5Real27"
```
The application will automatically load from environment variables first, falling back to local `telegram_config.json` only if environment variables are not set.
