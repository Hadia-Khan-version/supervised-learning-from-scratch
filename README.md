# Supervised Learning from Scratch: Regression, Classification & Regularization

An interactive walkthrough of core supervised learning concepts — linear regression, polynomial regression, logistic regression, underfitting/overfitting, and regularization (Ridge & Lasso) — built to explain the *why*, not just run the code. Two of the foundational algorithms (linear regression, logistic regression) are implemented from scratch with manual gradient descent and then verified against scikit-learn.

**Run the interactive version here:** [Kaggle Notebook](https://www.kaggle.com/code/hadiaaakhaaan/supervised-learning-from-scratch)

> GitHub renders notebooks statically, so sliders and dropdowns won't respond here — for the full interactive experience (degree sliders, threshold sliders, decision boundary explorers), use the Kaggle link above or run it locally.

## What's covered

1. **Linear Regression** — hypothesis function, cost function, gradient descent implemented from scratch and compared against sklearn, with a live learning-rate/iteration explorer
2. **Polynomial Regression, Underfitting & Overfitting** — fitting curves via feature transformation, a degree slider showing the underfit → good fit → overfit spectrum, and a train/validation error curve
3. **Regularization (Ridge & Lasso)** — L1 vs L2 penalties, an alpha slider showing coefficients shrink (Ridge) or zero out (Lasso), tied back to the bias-variance tradeoff
4. **Logistic Regression** — sigmoid, log loss, and decision boundaries, first on clean synthetic blobs (from-scratch gradient descent + sklearn comparison), then on the real Breast Cancer Wisconsin dataset with a feature-pair explorer, threshold/confusion-matrix widget, and ROC curve
5. **Wrap-up** — a recap table of when to use each model, and an interactive scenario-based quiz to test intuition

## Datasets used

All built into scikit-learn — no downloads needed beyond the library itself:
- California Housing (linear regression)
- Synthetic sine wave with noise (polynomial regression, regularization)
- Synthetic blobs via `make_blobs` (logistic regression, stage A)
- Breast Cancer Wisconsin (logistic regression, stage B)

## Running locally

```bash
pip install -r requirements.txt
jupyter notebook
```

Open `notebook.ipynb` and run all cells. Widgets require `ipywidgets` to be enabled in your Jupyter environment.

## License

Released under the [MIT License](LICENSE).
