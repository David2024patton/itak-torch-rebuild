# Glimmer accuracy spot check prompts (fixed set, temp 0, max_tokens 512)
# Reference = Q5_K_M output on the faster backend. Scored: exact/codable checks + diff vs reference.

1. MATH (exact answer check: 42)
"A store sells pencils in packs of 6 for $4.50 per pack. Maria needs 84 pencils and has a coupon for 15% off her total. How much does she pay? Show your work step by step, then give the final amount."

2. GO_CODE (compiles-or-not + does it sort)
"Write a Go function `MedianOfTwoSortedArrays(a, b []int) float64` that returns the median of two sorted slices in O(log(min(n,m))) time. Code only, no explanation."

3. LOGIC (exact answer check)
"Three boxes are labeled 'Apples', 'Oranges', and 'Apples & Oranges', but every label is wrong. You may draw one fruit from one box. Which box do you draw from, and how do you then correctly label all three? Answer concisely."

4. JSON_FORMAT (valid JSON, exact keys)
"Extract the data and reply with ONLY a JSON object, no other text: 'Jane Cooper, age 34, ordered 3 units of SKU-2214 on March 3rd, 2026, total $129.99, shipping to Austin, TX.' Keys: name, age, units, sku, date (ISO format), total, city, state."

5. FACTUAL (known answer)
"In Ursula K. Le Guin's novel The Left Hand of Darkness, what is the name of the planet where most of the story takes place, what are its people called, and what is unusual about their biology? Two or three sentences."

6. SUMMARY (3 bullets, diff vs reference)
"Summarize in exactly 3 bullet points: Continuous batching lets an inference server swap finished requests out and new ones in at every token step instead of waiting for a whole batch. This raises GPU utilization because short requests no longer idle behind long ones. The cost is scheduler complexity and the need for paged KV cache management so sequences can start and stop without fragmenting memory. Operators also gain per-request controls like max tokens and stop sequences, which batch-at-once systems handle poorly."
