# End-to-End LLM Tests Plan

## Overview

Create comprehensive end-to-end tests with real LLM calls to ensure the async engine works correctly in production.

## Issues Identified

1. **AsyncPersonaGenerator**: Only generates random personas, doesn't call LLM
2. **AsyncInterviewProcessor**: Incomplete prompt generation
3. **Missing Real LLM Integration**: Tests are mocked, production isn't calling LLM
4. **No End-to-End Validation**: No tests verify actual LLM responses

## Test Sections

### 1. Real LLM Integration Tests

- Test AsyncPersonaGenerator with real LLM calls
- Test AsyncInterviewProcessor with real LLM calls
- Test complete interview flow with real responses
- Test rate limiting and concurrency with real API calls

### 2. Prompt Generation Tests

- Test that prompts are generated correctly for personas
- Test that prompts are generated correctly for interviews
- Test that prompts include all necessary context
- Test that prompts are unique across cycles

### 3. Response Parsing Tests

- Test parsing of real LLM responses
- Test extraction of insights from responses
- Test alignment detection from responses
- Test pain point extraction from responses

### 4. End-to-End Flow Tests

- Test complete cycle with real LLM calls
- Test multiple cycles with evolution
- Test that insights are properly aggregated
- Test that reports are generated with real data

### 5. Error Handling Tests

- Test API rate limiting
- Test API errors and retries
- Test malformed responses
- Test network timeouts

### 6. Production Validation Tests

- Test with actual YGT config
- Test with real OpenAI API
- Test complete 3-cycle run
- Validate output quality and insights

## Implementation Steps

1. Fix AsyncPersonaGenerator to call LLM
2. Fix AsyncInterviewProcessor prompt generation
3. Create end-to-end tests with real LLM calls
4. Add proper error handling and retries
5. Test with actual production config
6. Validate output quality

## Success Criteria

- AsyncPersonaGenerator calls LLM to generate personas
- AsyncInterviewProcessor calls LLM for interviews
- Real insights are extracted from LLM responses
- End-to-end tests pass with real API calls
- Production run generates meaningful insights
- Rate limiting and concurrency work correctly
