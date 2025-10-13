# Machine Learning User Guide

## What is Machine Learning?

**Machine Learning (ML)** is a branch of **Artificial Intelligence (AI)** that enables computers to **learn from data** and make **predictions or decisions** without being explicitly programmed.  

For example:  
``Your email’s spam filter “learns” to detect spam by identifying patterns from past messages.``

---

## Types of Machine Learning

| Type | Description | Example |
|------|--------------|----------|
| **Supervised Learning** | Learns from labeled data — the model is trained on input-output pairs. | Predicting house prices |
| **Unsupervised Learning** | Discovers hidden patterns in unlabeled data. | Grouping customers into clusters |
| **Reinforcement Learning** | Learns through trial and error to maximize rewards. | Training self-driving cars |

---

## Example

Here’s a simple machine learning example using **scikit-learn**:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```
This trains a linear regression model using training data ``(X_train, y_train)`` and then generates predictions from new data ``(X_test)``.

## FAQs

**Q: What’s the difference between AI and ML?**

A: AI is the broader concept of machines simulating intelligence. ML is a subset focused on learning from data through algorithms.

**Q: Do I need math to learn ML?**

A: A foundation in algebra, statistics, and probability helps — but modern tools like scikit-learn, TensorFlow, and PyTorch make ML accessible to everyone.

**Q: Can ML make mistakes?**

A: Yes. Models are only as accurate as the data they’re trained on. Bias, poor data quality, or lack of context can all lead to errors in predictions.

## In summary:
Machine Learning empowers computers to recognize patterns, adapt over time, and make data-driven decisions — shaping everything from recommendation systems to self-driving cars.