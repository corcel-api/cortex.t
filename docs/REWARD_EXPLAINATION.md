# Cortex.t Reward System Documentation

## Platform Overview

Cortex.t provides decentralized infrastructure for running closed-source APIs, focusing on two primary services: Text Generation and Image Generation. The platform implements a sophisticated reward mechanism where validators evaluate miners' performance to determine appropriate compensation. Each service type employs distinct validation methodologies and scoring systems to ensure fair and accurate reward distribution.

## Text Generation Service

### Model Support and Implementation

The text generation service currently supports two major API providers:

1. OpenAI Models (including `gpt-4`)
2. Anthropic Models (including `claude-3-5-sonnet-20241022`)

Both providers are integrated with deterministic response capabilities through seed inputs, ensuring consistent validation results.

### Validation Architecture

The validation system operates through a multi-step process designed to ensure accuracy and fairness:

1. Response Generation
   - Validators regenerate responses using identical input parameters as the miner's request
   - Response parameters and conditions are exactly matched to ensure fair comparison
   - Seed values ensure deterministic outputs for consistent evaluation

2. Similarity Assessment
   - Response similarity is computed between miner and validator outputs
   - OpenAI's text embedding model is used for comparison
   - Similarity scores directly influence reward allocation

3. Resource Optimization
   - Validators use a predetermined portion of miner-committed credits for synthetic scoring
   - Validation occurs once per epoch rather than per response
   - This approach balances accuracy with resource efficiency

### Validation Process Implementation

The system implements a sophisticated batch processing approach:

1. Batch Processing Architecture
   - Miners are processed in configurable batch sizes
   - Multiple concurrent batches operate simultaneously
   - Each batch contains an optimized number of miners for efficient processing

2. Response Validation Flow
   The system follows a structured validation sequence:
   ```pseudo
   For each batch:
       synthetic_prompt = generate_test_prompt()
       responses = query_miners(miners, synthetic_prompt)
       
       For each response in responses:
           if response.is_success and response.verify():
               valid_responses.append(response)
           else:
               score_miner(miner_id, score=0)
   ```

3. Score Calculation
   Responses are scored using a comprehensive formula:
   ```pseudo
   For each valid_response:
       base_score = calculate_similarity(valid_response, synthetic_prompt)
       time_penalty = response.process_time / response.timeout
       final_score = max(base_score - (time_penalty * 0.2), 0)
       score_miner(miner_id, final_score)
   ```

### Scoring Components

The scoring system incorporates four key components:

1. Base Similarity Score
   - Utilizes embedding similarity between miner's response and validator's synthetic prompt
   - Implemented through OpenAI's text embedding model
   - Higher similarity correlates with higher base scores

2. Time Penalty Calculation
   - Penalty = response.process_time / response.timeout
   - Maximum penalty capped at 20% of base score
   - Final score = max(base_score - (time_penalty * 0.2), 0)

3. Credit Scale Implementation
   - Scale = min(miner_credit / max_credit, 1.0)
   - Final weighted score = score * credit_scale
   - Ensures proportional reward based on credit commitment

4. Score Accumulation Method
   - Uses Exponential Moving Average (EMA) with configurable decay factor
   - Formula: previous_score * decay_factor + new_score * (1 - decay_factor)
   - Provides score stability over time

## Image Generation Service

### Model Support
Currently supports OpenAI's `dall-e-3` model with potential for future expansion.

### Validation Methodology

The image validation process follows three distinct steps:

1. URL Pattern Validation
   - Verifies returned image URL matches OpenAI's expected pattern
   - Failed pattern validation results in zero score

2. Metadata Authentication
   - Examines image metadata/EXIF data for `dall-e-3` compatibility
   - Metadata verification failure results in zero score

3. Content Analysis
   - Evaluates image-prompt alignment
   - Assigns similarity score based on content matching accuracy

### Efficiency Considerations
The image service achieves higher resource efficiency by eliminating the need for response regeneration during validation.

## Optimization Guidelines for Miners

To maximize rewards, miners should focus on:

1. Response Quality Optimization
   - Ensure high accuracy and quality in responses
   - Maintain proper formatting and verification standards
   - Understand that failed verifications result in immediate zero scores

2. Performance Timing
   - Optimize response speed to minimize time penalties
   - Process requests well within timeout thresholds
   - Account for hardware capabilities and network latency

3. Credit Management Strategy
   - Maintain optimal credit balance (minimum: CONFIG.bandwidth.max_credit)
   - Monitor and manage credit consumption rates
   - Understand credit scaling impact on rewards

4. System Availability
   - Maintain consistent online presence
   - Handle concurrent requests efficiently
   - Maximize scoring opportunities (4 per epoch)
   - Align with 10-minute weight update cycles

## Timing Parameters

The validation system operates on specific timing parameters:

1. Epoch Limitations
   - Maximum 4 scores per miner per epoch
   - Score records expire after 360 seconds
   - Regular validation cycles ensure fair evaluation

2. Weight Update Schedule
   - Network weights updated every 600 seconds (10 minutes)
   - Allows for score accumulation before adjustments
   - Provides stability in reward distribution

3. Processing Optimization
   - Configurable batch sizes for efficient processing
   - Multiple concurrent batches for throughput optimization
   - Synthetic thresholds ensure minimum quality standards