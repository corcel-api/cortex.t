<div align="left">

# **Cortex.t Subnet** <!-- omit in toc -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
---

---
- [Introduction](#introduction)
- [Setup](#setup)
- [Mining](#mining)
- [Validating](#validating)
- [License](#license)


## Introduction

**IMPORTANT**: If you are new to Bittensor, please checkout the [Bittensor Website](https://bittensor.com/) before proceeding to the [Setup](#setup) section.

Introducing Bittensor Subnet 18 (Cortex.t): A Pioneering Platform for AI Development and Synthetic Data Generation.

Cortex.t stands at the forefront of artificial intelligence, offering a dual-purpose solution that caters to the needs of app developers and innovators in the AI space. This platform is meticulously designed to deliver reliable, high-quality text and image responses through API usage, utilising the decentralised Bittensor network. It serves as a cornerstone for creating a fair, transparent, and manipulation-free environment for the incentivised production of intelligence (mining) and generation and fulfilment of diverse user prompts.

Our initiative is a leap forward in redefining the reward system for text and image prompting with a commitment to providing stability and reassurance to developers. By focusing on the value delivered to clients, we alleviate the concerns of data inconsistencies that often plague app development. The quality of Cortex.t is seamlessly integrated within the Bittensor network, allowing developers to harness the power of multiple subnets and modalities by building directly onto an existing validator, or through an API key from [Corcel](https://corcel.io).

Cortex.t is also a transformative platform leveraging advanced AI models to generate synthetic prompt-response pairs. This novel method yields a comprehensive dataset of interactions, archived in wandb [wandb.ai/cortex-t/synthetic-QA](https://wandb.ai/cortex-t/synthetic-QA). The process involves recycling model outputs back into the system, using a prompt evolution and data augmentation strategy similar to Microsoft's approach in developing WizardLM. This enables the distillation of sophisticated AI models into smaller, yet efficient counterparts, mirroring the performance of their larger predecessors. Ultimately, Cortex.t democratizes access to high-end AI technology, encouraging innovation and customization.

By leveraging synthetic data, Cortex.t circumvents the traditional challenges of data collection and curation, accelerating the development of AI models that are both robust and adaptable. This platform is your gateway to AI mastery, offering the unique opportunity to train your models with data that reflects the depth and versatility of the parent model. With SynthPairPro, you're not just collecting data; you're capturing intelligence, providing a path to creating AI models that mirror the advanced understanding and response capabilities of their predecessors.

Join us at Cortex.t, your bridge to AI excellence, and democratise access to top-level AI capabilities. Be part of the AI revolution and stay at the forefront of innovation with SynthPairPro – Synthesizing Intelligence, Empowering the Future!


## Setup

### Before you proceed
Before you proceed with the installation of the subnet, note the following:

**IMPORTANT**: We **strongly recommend** before proceeding that you test both subtensor and all API keys. Ensure you are running Subtensor locally to minimize chances of outages and improve the latency/connection.

After exporting your OpenAI API key to your bash profile, test the streaming service for both the gpt-3.5-turbo and gpt-4 engines using ```./neurons/test_openai.py```. Neither the miner or the validator will function without a valid and working [OpenAI API key](https://platform.openai.com/).

**IMPORTANT:** Make sure you are aware of the minimum compute requirements for cortex.t. See the [Minimum compute YAML configuration](./min_compute.yml).
Note that this subnet requires very little compute. The main functionality is API calls, so we outsource the compute to the providors of these keys. The cost for mining and validating on this subnet comes from API calls, not from compute. Please be aware of your API costs and monitor accordingly.

A high tier key is required for both mining and validations so it is important if you do not have one to work your way up slowly by running a single miner or small numbers of miners whilst paying attention to your usage and limits.

### Requesting Access for AWS Bedrock Models

#### 1. AWS Account
Ensure you have an active AWS account. If you don't have one, you can create it at [AWS Account Creation](https://aws.amazon.com/).

#### 2. Sign In to AWS Management Console
Go to the [AWS Management Console](https://aws.amazon.com/console/) and sign in with your AWS credentials.

#### 3. Navigate to AWS Bedrock
- In the AWS Management Console, use the search bar at the top to search for "Bedrock".
- Select AWS Bedrock from the search results.

#### 4. Request Access
- If AWS Bedrock is not directly available, you might see a page to request access.
- Follow the prompts to fill out any required information. This might include your use case for the models, your AWS account ID, and other relevant details.

#### 5. Submit a Request
- Complete any forms or questionnaires provided to describe your intended use of AWS Bedrock models.
- Submit the request for review.

#### 6. Wait for Approval
- AWS will review your request. This can take some time depending on the specifics of your request and the current availability of AWS Bedrock.
- You will receive an email notification once your request is approved or if further information is needed.

### Obtaining AWS Access Key and Secret Key

#### 1. Sign In to AWS Management Console
Go to the [AWS Management Console](https://aws.amazon.com/console/) and sign in with your AWS credentials.

#### 2. Navigate to My Security Credentials
- Click on your account name at the top right corner of the AWS Management Console.
- Select "Security Credentials" from the dropdown menu.

#### 3. Create New Access Key
- In the "My Security Credentials" page, go to the "Access keys" section.
- Click on "Create Access Key".
- A pop-up will appear showing your new Access Key ID and Secret Access Key.

#### 4. Download Credentials
- Download the `.csv` file containing these credentials or copy them to a secure location.
  - **Important**: This is the only time you will be able to view the secret access key. If you lose it, you will need to create new credentials.


### Obtaining API Key from OpenAI

#### 1. OpenAI Account
Ensure you have an active OpenAI account. If you don't have one, you can create it at [OpenAI Account Creation](https://platform.openai.com/signup).

#### 2. Sign In to OpenAI
Go to the [OpenAI Platform](https://platform.openai.com/api-keys) and sign in with your OpenAI credentials.

#### 3. Create New API Key
- Click on the "Create new secret key" button.
- Follow the instructions provided to create your API key.


### Obtaining API Key from Google AI Platform

#### 1. Sign In to Google AI Platform
Go to the [Google AI Platform](https://aistudio.google.com/) and sign in with your Google credentials.

#### 2. Get API Key
- In the Google AI Platform, click on the "Get API key" button at the top left corner.
- Follow the instructions provided to create and retrieve your API key.


### Obtaining API Key from Anthropic

#### 1. Anthropic Account
Ensure you have an active Anthropic account. If you don't have one, you can create it at [Anthropic Account Creation](https://www.anthropic.com/signup).

#### 2. Sign In to Anthropic
Go to the [Anthropic Platform](https://console.anthropic.com/settings/keys) and sign in with your Anthropic credentials.

#### 3. Get API Key
- In the Settings, go to the "API keys" tab and click on the "Create key" button at the top right corner.
- Follow the instructions provided to create and retrieve your API key.


### Obtaining API Key from Groq

#### 1. Groq Account
Ensure you have an active Groq account. If you don't have one, you can create it at [Groq Account Creation](https://groq.com/signup).

#### 2. Sign In to Groq
Go to the [Groq Platform](https://console.groq.com/) and sign in with your Groq credentials.

#### 3. Get API Key
- In the Groq Platform, click on the "API keys" button at the left side.
- Click "Create API key"
- Follow the instructions provided to create and retrieve your API key.


### Obtaining API Key from Pixabay

#### 1. Pixabay Account
Ensure you have an active Pixabay account. If you don't have one, you can create it at [Pixabay Account Creation](https://pixabay.com/ru/accounts/register/).

#### 2. Sign In to Pixabay
Go to the [Pixabay API docs](https://pixabay.com/api/docs/) and sign in with your Pixabay credentials.

#### 3. Get API Key
- Scroll down this page a little. Your key will be highlighted in green in the parameters for one of the requests.


### API Key Requirements

API requirements for this subnet are constantly evovling as we seek to meet the demands of the users and stay up to date with the latest developements. The current key requirements are as follows:

- OpenAI key (GPT)
- Google API key (Gemini)
- Anthropic API key (Claude3)
- Groq API key (Llama, Gemini, Mistral)
- AWS Acces key and Secret key (Bedrock models)
- Pixabay API key


The higher rate limit your key has the better, and it can be advisable if mining to build up your rate limit slowly (even starting on testnet) to maximise your chances of achieving optimum performance.

### Installation

Before starting make sure you have pm2, nano and any other useful tools installed.

```bash
apt update -y && apt-get install git -y && apt install python3-pip -y && apt install npm -y && npm install pm2@latest -g  && apt install nano
```

Download the repository, navigate to the folder and then install the necessary requirements with the following chained command.

```bash
git clone https://github.com/corcel-api/cortex.t.git && cd cortex.t && pip install -e .
```

Prior to proceeding, ensure you have a registered hotkey on subnet 18 mainnet. If not, run the command
```bash
btcli s register --netuid 18 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey]
```

We recommend using [direnv](https://direnv.net). After installing it, copy `envrc.example` to `.envrc` and substitute
all env vars with values appropriate for your accounts. After making changes to `.envrc` run `direnv allow` and start a
new terminal tab.

## Mining

You can launch your miners via pm2 using the following command.

```bash
pm2 start ./miner/miner.py --interpreter python3 -- --netuid 18 --subtensor.network <LOCAL/FINNEY/TEST> --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME> --axon.port <PORT>
```


## Validating

Login to wandb using

```bash
wand login
```

You can launch your validator via pm2 WITH AUTO-UPDATES (reccomended) using the following command.

```bash
pm2 start ./start_validator.py --interpreter python3 -- --netuid 18 --subtensor.network <LOCAL/FINNEY/TEST> --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME>
```

You can launch your validator via pm2 WITHOUT AUTO-UPDATES using the following command.

```bash
pm2 start ./validators/validator.py --interpreter python3 -- --netuid 18 --subtensor.network <LOCAL/FINNEY/TEST> --wallet.name <WALLET NAME> --wallet.hotkey <HOTKEY NAME>
```

## Logging

As cortex.t supports streaming natively, you do not (and should not) enable `logging.trace` or `logging.debug` as all of the important information is already output to `logging.info` which is set as default.

---

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
