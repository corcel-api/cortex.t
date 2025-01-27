# Cortex.t Reward Functional Documentation

## Overview

Cortex.t provides a decentralized infrastructure for closed-source APIs, primarily focusing on two main services:
- Text Generation
- Image Generation

The platform implements a reward mechanism where validators evaluate miners' responses to determine appropriate compensation. Each service type has its own validation methodology and scoring system.

## Text Generation Service

### Supported Models
- OpenAI Models (e.g., `gpt-4`)
- Anthropic Models (e.g., `claude-3-5-sonnet-20241022`)

### Validation Process
1. Validators regenerate responses using identical input parameters as the miner's request
2. Response similarity is computed between the miner's and validator's outputs
3. Rewards are allocated based on the similarity score, using openai text embedding model
4. Deterministic responses are enabled through seed inputs for both OpenAI and Anthropic models

### Resource Management
- Validators consume a predetermined maximum portion of credits committed by miners for synthetic scoring
- To optimize resource usage, validation occurs once per epoch rather than for every response
- Note: While this approach involves duplicate API calls, the once-per-epoch validation system helps maintain efficiency

## Image Generation Service

### Supported Models
- OpenAI Models (e.g., `dall-e-3`)

### Validation Process
The validator performs a three-step verification:

1. URL Pattern Validation
   - Checks if the returned image URL matches the expected pattern of OpenAI
   - Returns a score of 0 if pattern validation fails

2. Metadata Verification
   - Examines image metadata/EXIF data for compatibility with the `dall-e-3` model
   - Returns a score of 0 if metadata verification fails

3. Content Evaluation
   - Assesses whether the generated image accurately depicts the input prompt
   - Assigns a similarity score based on content matching

The final reward is calculated based on the cumulative results of these verification steps.

### Resource Efficiency
Unlike the text generation service, image validation does not require regenerating the content, making it more resource-efficient.

<iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="800" height="450" src="https://embed.figma.com/board/1ikEvDEIS2zgQGNGcmL474/Untitled?node-id=0-1&embed-host=share" allowfullscreen></iframe>