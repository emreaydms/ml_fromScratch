# Machine Learning Algorithms from Scratch 

A comprehensive implementation of fundamental machine learning algorithms and optimization techniques built entirely from scratch using NumPy. This project demonstrates deep understanding of ML fundamentals without relying on high-level libraries like scikit-learn.

## Key Features

### Optimization Algorithms
- **Zero-order methods**: Random Search, Coordinate Search
- **First-order methods**: Gradient Descent, Coordinate Descent  
- **Second-order methods**: Newton's Method
- **Advanced features**: Diminishing learning rates, convergence criteria

### Linear Regression Suite
- **Least Squares** - Closed-form solution using normal equations
- **Ridge Regression** - L2 regularization with analytical solution
- **Lasso Regression** - L1 regularization with subgradient descent
- **Elastic Net** - Combined L1/L2 regularization

### Model Selection & Evaluation
- **Custom Cross-Validation**: K-fold implementation from scratch
- **Hyperparameter Tuning**: Randomized search with custom parameter distributions
- **Performance Metrics**: MSE, R² score implementations
- **Data Preprocessing**: MinMaxScaler, train-test splitting utilities

## Technical Highlights

- **Mathematical Rigor**: Implements core algorithms using fundamental mathematical principles
- **Numerical Stability**: Uses pseudo-inverse and proper regularization techniques
- **Modular Design**: Clean OOP architecture with inheritance and polymorphism
- **Performance Optimization**: Efficient NumPy vectorization throughout
- **Comprehensive Testing**: Validated against analytical solutions and numerical gradients

## Real-World Applications

### Boston Housing Price Prediction
Complete end-to-end ML pipeline demonstrating:
- Data preprocessing and feature scaling
- Model comparison across multiple algorithms
- Hyperparameter optimization
- Cross-validation and performance evaluation
- Visualization of predictions vs. actual values

### Optimization Visualization
Interactive contour plots showing convergence paths for different optimization algorithms, providing intuitive understanding of algorithm behavior.

## Technical Stack

- **Core**: Python, NumPy
- **Visualization**: Matplotlib
- **Mathematical Computing**: Linear algebra, numerical optimization
- **Design Patterns**: Object-oriented programming, inheritance hierarchies

## Project Structure

```
├── linear_regression.py    # Complete regression algorithm suite
├── optim.py               # Optimization algorithms collection
├── metrics.py             # Performance evaluation metrics
├── search.py              # Hyperparameter tuning framework
├── utils.py               # Data preprocessing utilities
└── notebook.ipynb        # Comprehensive demonstrations
```

## Learning Outcomes

This project showcases proficiency in:

- **Mathematical Foundation**: Deep understanding of optimization theory and linear algebra
- **Algorithm Implementation**: Ability to translate mathematical concepts into efficient code
- **Software Engineering**: Clean, modular, and maintainable code architecture
- **Data Science Pipeline**: Complete ML workflow from data preprocessing to model evaluation
- **Problem Solving**: Debug and optimize numerical algorithms for stability and performance

## Code Quality Features

- **Comprehensive Documentation**: Detailed docstrings for all methods
- **Error Handling**: Robust input validation and edge case management
- **Type Hints**: Clear function signatures with proper typing
- **Modular Design**: Easily extensible and reusable components
- **Performance Monitoring**: Runtime analysis and convergence tracking

## Getting Started

```python
# Quick example: Ridge Regression
from linear_regression import RidgeRegression
from metrics import r2_score

# Initialize and train model
model = RidgeRegression(lambd=1.0)
model.fit(X_train, y_train)

# Make predictions and evaluate
predictions = model.predict(X_test)
score = r2_score(y_true, predictions)
```

## Why This Matters

Building ML algorithms from scratch demonstrates:
- **Deep Technical Understanding** beyond using pre-built libraries
- **Mathematical Proficiency** in optimization and statistical learning
- **Problem-Solving Skills** in numerical computing and algorithm design
- **Code Quality** standards suitable for production environments

---

*This project represents a solid foundation in machine learning fundamentals, showcasing the ability to understand, implement, and optimize core algorithms that power modern ML systems.*
