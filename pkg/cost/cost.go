package cost

import (
	"strings"
	"sync"
)

// PricingModel defines the cost per 1M tokens (standard industry pricing unit)
type PricingModel struct {
	InputPrice  float64 // Cost per 1M input tokens
	OutputPrice float64 // Cost per 1M output tokens
}

// CostCalculator calculates estimated costs for LLM usage
type CostCalculator struct {
	prices map[string]PricingModel
	mu     sync.RWMutex
}

// NewCostCalculator creates a new calculator with default pricing
func NewCostCalculator() *CostCalculator {
	c := &CostCalculator{
		prices: make(map[string]PricingModel),
	}
	c.loadDefaults()
	return c
}

// CalculateCost returns the estimated cost in USD
func (c *CostCalculator) CalculateCost(model string, promptTokens, completionTokens int) float64 {
	c.mu.RLock()
	price, ok := c.prices[strings.ToLower(model)]
	if !ok {
		// Fallback to generic high-end pricing (gpt-4o) if unknown, or maybe 0?
		// Let's try to find a partial match or default to 0
		price = PricingModel{0, 0}

		// Heuristic: Check for common prefixes
		if strings.HasPrefix(strings.ToLower(model), "gpt-4") {
			price = c.prices["gpt-4o"]
		} else if strings.HasPrefix(strings.ToLower(model), "gpt-3.5") {
			price = c.prices["gpt-3.5-turbo"]
		} else if strings.HasPrefix(strings.ToLower(model), "claude-3-opus") {
			price = c.prices["claude-3-opus"]
		} else if strings.HasPrefix(strings.ToLower(model), "claude-3-sonnet") {
			price = c.prices["claude-3-sonnet"]
		} else if strings.HasPrefix(strings.ToLower(model), "claude-3-haiku") {
			price = c.prices["claude-3-haiku"]
		}
	}
	c.mu.RUnlock()

	inputCost := (float64(promptTokens) / 1_000_000.0) * price.InputPrice
	outputCost := (float64(completionTokens) / 1_000_000.0) * price.OutputPrice

	return inputCost + outputCost
}

// loadDefaults loads standard pricing for major providers (as of late 2024/early 2025)
func (c *CostCalculator) loadDefaults() {
	// OpenAI
	c.prices["gpt-4o"] = PricingModel{InputPrice: 2.50, OutputPrice: 10.00}
	c.prices["gpt-4o-mini"] = PricingModel{InputPrice: 0.15, OutputPrice: 0.60}
	c.prices["gpt-4-turbo"] = PricingModel{InputPrice: 10.00, OutputPrice: 30.00}
	c.prices["gpt-3.5-turbo"] = PricingModel{InputPrice: 0.50, OutputPrice: 1.50} // Legacyish
	c.prices["o1-preview"] = PricingModel{InputPrice: 15.00, OutputPrice: 60.00}
	c.prices["o1-mini"] = PricingModel{InputPrice: 3.00, OutputPrice: 12.00}

	// Anthropic
	c.prices["claude-3-5-sonnet"] = PricingModel{InputPrice: 3.00, OutputPrice: 15.00}
	c.prices["claude-3-opus"] = PricingModel{InputPrice: 15.00, OutputPrice: 75.00}
	c.prices["claude-3-sonnet"] = PricingModel{InputPrice: 3.00, OutputPrice: 15.00}
	c.prices["claude-3-haiku"] = PricingModel{InputPrice: 0.25, OutputPrice: 1.25}

	// Default/Fallback mappings
	c.prices["gpt-4"] = c.prices["gpt-4o"]
	c.prices["unknown"] = PricingModel{0, 0}

	// Together AI Pricing (Serverless Inference) - https://www.together.ai/pricing

	// Llama 3.3 70B
	c.prices["meta-llama/llama-3.3-70b-instruct-turbo"] = PricingModel{InputPrice: 0.88, OutputPrice: 0.88}

	// Llama 3.2
	c.prices["meta-llama/llama-3.2-3b-instruct-turbo"] = PricingModel{InputPrice: 0.06, OutputPrice: 0.06}
	c.prices["meta-llama/llama-3.2-11b-vision-instruct-turbo"] = PricingModel{InputPrice: 0.18, OutputPrice: 0.18}
	c.prices["meta-llama/llama-3.2-90b-vision-instruct-turbo"] = PricingModel{InputPrice: 1.20, OutputPrice: 1.20}

	// Llama 3.1
	c.prices["meta-llama/meta-llama-3.1-8b-instruct-turbo"] = PricingModel{InputPrice: 0.18, OutputPrice: 0.18}
	c.prices["meta-llama/meta-llama-3.1-70b-instruct-turbo"] = PricingModel{InputPrice: 0.88, OutputPrice: 0.88}
	c.prices["meta-llama/meta-llama-3.1-405b-instruct-turbo"] = PricingModel{InputPrice: 5.00, OutputPrice: 15.00}

	// Qwen 2.5
	c.prices["qwen/qwen2.5-7b-instruct-turbo"] = PricingModel{InputPrice: 0.30, OutputPrice: 0.30}
	c.prices["qwen/qwen2.5-72b-instruct-turbo"] = PricingModel{InputPrice: 1.20, OutputPrice: 1.20}
	c.prices["qwen/qwen2.5-coder-32b-instruct"] = PricingModel{InputPrice: 0.80, OutputPrice: 0.80}

	// Mixtral
	c.prices["mistralai/mixtral-8x7b-instruct-v0.1"] = PricingModel{InputPrice: 0.60, OutputPrice: 0.60}
	c.prices["mistralai/mixtral-8x22b-instruct-v0.1"] = PricingModel{InputPrice: 1.20, OutputPrice: 1.20}

	// DeepSeek
	c.prices["deepseek-ai/deepseek-v3"] = PricingModel{InputPrice: 1.25, OutputPrice: 1.25}
}
