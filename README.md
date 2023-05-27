# OpenAI Telegram Bot

This is a repository for a chatbot built using OpenAI's chat completion API. It allows users to ask the bot questions and receive human-like responses from the AI model.

## Prerequisites

1. [Get OpenAI API key](https://platform.openai.com/account/api-keys).
2. [Get Telegram bot token](https://core.telegram.org/bots/tutorial#obtain-your-bot-token).
3. [Install Docker](https://docs.docker.com/engine/install/).

   Here is command for installation on Debian/Ubuntu via convenience script:

   ```bash
   sudo apt update && \
   curl -fsSL https://get.docker.com -o get-docker.sh && \
   /bin/bash ./get-docker.sh && \
   sudo apt install -y docker-compose
   ```

## How to run locally

1. Clone this repo onto your local machine:

   ```bash
   git clone https://github.com/Danand/bot-openai.git
   ```

2. Create `.env` file with the necessary variables:

   ```bash
   LOG_LEVEL="DEBUG"

   TELEGRAM_API_TOKEN="<your-token-here>"
   TELEGRAM_WHITELISTED_USERS="<telegram-user-id-integer-of-admins-comma-separated>"

   OPENAI_API_KEY="<your-token-here>"
   OPENAI_DEFAULT_MODEL="gpt-3.5-turbo"
   OPENAI_DEFAULT_TEMPERATURE="1.0"
   OPENAI_DEFAULT_MAX_TOKENS="0"
   OPENAI_DEFAULT_MAX_MESSAGES="5"

   REDIS_HOST="bot-openai-redis"
   REDIS_PORT=6380
   REDIS_VERSION="7.0.11-alpine3.18"
   ```

3. Run the bot using the command:

   ```bash
   docker-compose \
     -f ./docker-compose.yml \
     -f ./docker-compose-local.yml \
     up
   ```

## How to deploy

1. Fork this repo to your own account.
2. Set up necessary secrets in your secrets settings of your repo:

   - `DEPLOY_SSH_KEY`
   - `HOST_PUBLIC_IP`
   - `OPENAI_API_KEY`
   - `TELEGRAM_API_TOKEN`
   - `TELEGRAM_WHITELISTED_USERS`
   - `_GITHUB_PAT`

3. Make changes if you want to.
4. Make sure you have compatible [GitHub Actions Runner](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners) for this repo.
5. Push new tag, workflow **Build and Push Docker Images** will be triggered automatically.
6. Run workflow **Deploy Docker Images** manually with chosen tag to deploy image built at previous step.
