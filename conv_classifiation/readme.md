# Signal Extraction Benchmark

## Restriction

## Dataset

Since no labeled dataset was provided for the task, a synthetic dataset was created to simulate realistic VIP casino guest conversations and evaluate the signal extraction system.

The goal of the dataset is to reproduce the types of conversational signals that a casino host or concierge system might encounter when interacting with high-value guests.

These signals include:

- Intent (e.g., planning a trip or booking a room)
- Value indicators (e.g., preference for suites, high budget, group travel)
- Sentiment about past experiences
- Life events that may trigger special treatment (birthday, anniversary, promotion)
- Competitive signals mentioning other casinos

The dataset therefore represents a **multi-label structured information extraction problem** and was generated using the frontier model **GPT-5.3**.

---

## Infrastructure

Model benchmarking was performed through **OpenRouter**, which provides a unified API for accessing multiple LLM providers.

This allowed evaluating several frontier and open-source models using the same inference pipeline, ensuring consistent prompts, evaluation logic, and metrics across models.

## Model Selection

![Models benchmarks](models_benchmarks.png)

The goal of the model selection process was to identify models that offer **the best trade-off between cost and intelligence**, while also considering latency.

Based on benchmark performance and pricing constraints, the following models were evaluated:

- google/gemini-3.1-flash-lite-preview
- xiaomi/mimo-v2-flash
- deepseek/deepseek-v3.2
- x-ai/grok-4.1-fast

### Evaluation Metrics

The models were evaluated using the following metrics:

- **Precision**
- **Recall**
- **F1 Score**

---

## Results

![Results](results.png)

The **Gemini-3.1-flash-lite-preview model achieved the best performance** on the evaluation dataset.

A possible next step would be to integrate the model into an **agentic workflow**, allowing it to extract signals in real time from guest conversations and trigger downstream actions.