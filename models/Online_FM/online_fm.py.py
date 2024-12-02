# Databricks notebook source
from typing import List, Tuple

import joblib  # Import joblib for saving and loading the model
import numpy as np

class FMFTRL:
    def __init__(self, k, alpha=0.1, beta=1.0, l1=0.0, l2=0.0, join_key="="):
        self.alpha = alpha  # Learning rate
        self.beta = beta  # Regularization parameter
        self.l1 = l1  # L1 regularization
        self.l2 = l2  # L2 regularization
        self.k = k
        self.join_key = join_key
        print(f"Initialized with k: {self.k}")  # Debugging line

        # Initialize weights and factors
        self.W = {}  # Dictionary to hold weights
        self.V = {}  # Dictionary to hold factor weights
        self.z = {}  # Accumulated gradients for FTRL
        self.n = {}  # Accumulated squared gradients for FTRL

        # Initialize accumulated gradients for factor weights
        self.z_factors = {}  # Accumulated gradients for factor weights
        self.n_factors = {}  # Accumulated squared gradients for factor weights

        # Initialize bias term
        self.bias = 0.0
        self.z_bias = 0.0  # Accumulated gradient for bias
        self.n_bias = 0.0  # Accumulated squared gradient for bias

    def add_feature(self, feature_name):
        # Initialize weights and factors for the new feature
        if feature_name not in self.W:
            self.W[feature_name] = 0.0  # Initialize weight to 0

            # Debugging line to check k
            self.V[feature_name] = np.random.normal(
                scale=0.1, size=self.k
            )  # Initialize factors
            self.z[feature_name] = 0.0  # Initialize accumulated gradient
            self.n[feature_name] = 0.0  # Initialize accumulated squared gradient

            # Initialize accumulated gradients for factor weights
            self.z_factors[feature_name] = np.zeros(
                self.k
            )  # Initialize for each factor
            self.n_factors[feature_name] = np.zeros(
                self.k
            )  # Initialize for each factor

    def sigmoid(self, x):
        """Apply the numerically stable sigmoid function."""
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

    def binary_cross_entropy(self, y_true, y_pred):
        """Calculate binary cross-entropy loss."""
        y_true = np.array(y_true)
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def predict(self, batch_features: List[List[Tuple[int, float]]]) -> np.ndarray:
        """
        batch_features = ({'age_band': '20', 'gender': 'M', 'brand': 'a'}, {'age_band': '25', 'gender': 'W', 'brand': 'b'})
        """
        predictions = []
        for features in batch_features:
            linear_output = self.bias
            interactions = 0.0
            for feature_name, feature_value in features.items():
                if isinstance(feature_value, str):
                    feature_key = feature_name + self.join_key + str(feature_value)
                    update_value = 1.0
                else:
                    feature_key = feature_name
                    update_value = feature_value
                if feature_key not in self.W:
                    self.add_feature(feature_key)
                linear_output += self.W[feature_key] * update_value
                interactions += (
                    0.5
                    * (
                        np.dot(self.V[feature_key], self.V[feature_key])
                        - np.dot(self.V[feature_key], self.V[feature_key]) ** 2
                    )
                    * update_value
                )
            total_output = linear_output + interactions
            predictions.append(self.sigmoid(total_output))
        return np.array(predictions)

    def update(self, batch_features, batch_y):
        # Prediction
        y_pred = self.predict(batch_features)

        # Calculate loss
        loss = self.binary_cross_entropy(batch_y, y_pred)

        # Gradient calculation
        errors = batch_y - y_pred

        # Update weights and factors using FTRL
        for features, error in zip(batch_features, errors):
            # Update bias using FTRL
            self.z_bias += error - self.l1 * np.sign(self.bias) - self.l2 * self.bias
            self.n_bias += 1  # Increment the count for bias

            if self.n_bias > 0:
                self.bias = (self.z_bias - self.l1 * np.sign(self.bias)) / (
                    self.beta + np.sqrt(self.n_bias)
                )
            for feature_name, feature_value in features.items():
                if isinstance(feature_value, str):
                    feature_key = feature_name + self.join_key + str(feature_value)
                    update_value = 1.0
                else:
                    feature_key = feature_name
                    update_value = feature_value

                # Update accumulated gradients for linear weights
                self.z[feature_key] += (
                    error * update_value
                    - self.l1 * np.sign(self.W[feature_key])
                    - self.l2 * self.W[feature_key]
                )
                self.n[feature_key] += 1  # Increment the count for the feature

                # Update linear weight using FTRL
                if self.n[feature_key] > 0:
                    self.W[feature_key] = (
                        self.z[feature_key] - self.l1 * np.sign(self.W[feature_key])
                    ) / (self.beta + np.sqrt(self.n[feature_key]))

                # Update accumulated gradients for factor weights
                for f in range(self.k):
                    self.z_factors[feature_key][f] += (
                        error * self.V[feature_key][f] * update_value
                    )  # Accumulate gradient for factor f
                    self.n_factors[feature_key][
                        f
                    ] += 1  # Increment the count for the factor

                    # Update factor weights using FTRL
                    if self.n_factors[feature_key][f] > 0:
                        self.V[feature_key][f] = (
                            self.z_factors[feature_key][f]
                            - self.l1 * np.sign(self.V[feature_key][f])
                        ) / (self.beta + np.sqrt(self.n_factors[feature_key][f]))
        print(f"Loss: {loss}")
        return loss  # Return the loss

    def save_model(self, path):
        # Save the model as an artifact using joblib
        joblib.dump(self, path)

    def __call__(self, batch_features):
        return self.predict(batch_features)

    @classmethod
    def load_model(cls, path):
        # Load the model from an artifact
        return joblib.load(path)


class PandasDataset:
    def __init__(self, df, x_columns=None, y_column=None, is_train=True):
        self.df = df
        self.x_columns = x_columns
        self.y_column = y_column
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].values[0]
        x = {key: row[key] for key in self.x_columns}
        if self.is_train:
            y = row[self.y_column]
            return x, y
        else:
            return x


# COMMAND ----------



if __name__ == "__main__":

    import pandas as pd
    from torch.utils.data import DataLoader

    df = pd.DataFrame(
        [
            [
                {
                    "age_band": "20",
                    "gender": "M",
                    "brand": "a",
                    "click_cnt": 30,
                    "target": 1.0,
                }
            ],
            [
                {
                    "age_band": "25",
                    "gender": "W",
                    "brand": "b",
                    "click_cnt": 10,
                    "target": 0.0,
                }
            ],
            [
                {
                    "age_band": "30",
                    "gender": "M",
                    "brand": "c",
                    "click_cnt": 30,
                    "target": 1.0,
                }
            ],
            [
                {
                    "age_band": "20",
                    "gender": "M",
                    "brand": "a",
                    "click_cnt": 30,
                    "target": 1.0,
                }
            ],
        ]
    )

    dataset = PandasDataset(
        df,
        x_columns=["age_band", "gender", "brand", "click_cnt"],
        y_column="target",
        is_train=True,
    )
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=lambda x: list(zip(*x))
    )
    # d = next(iter(dataloader))

    model = FMFTRL(k=3)

    for batch in dataloader:
        features, y = batch
        print(f"features: {features}")
        features, y = list(features), list(y)

        model.update(features, y)

    # test
    model.save_model("fm_ftrl_model.joblib")
    model = FMFTRL.load_model("fm_ftrl_model.joblib")

    test_dataset = PandasDataset(
        df,
        x_columns=["age_band", "gender", "brand", "click_cnt"],
        y_column="target",
        is_train=False,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: list(x)
    )

    for test_batch in test_dataloader:
        features = test_batch
        features = list(features)
        predictions = model(features)
        print(predictions)
    print(1)


# COMMAND ----------

test=  [
            [
                {
                    "age_band": "20",
                    "gender": "M",
                    "brand": "a",
                    "click_cnt": 30,
                    "target": 1.0,
                }
            ],
            [
                {
                    "age_band": "25",
                    "gender": "W",
                    "brand": "b",
                    "click_cnt": 10,
                    "target": 0.0,
                }
            ],
            [
                {
                    "age_band": "30",
                    "gender": "M",
                    "brand": "c",
                    "click_cnt": 30,
                    "target": 1.0,
                }
            ],
            [
                {
                    "age_band": "20",
                    "gender": "M",
                    "brand": "a",
                    "click_cnt": 30,
                    "target": 1.0,
                }
            ],
        ]

# JSON 데이터를 평탄화(flatten)
flat_data = [item for sublist in test for item in sublist]

# DataFrame 생성
df = pd.DataFrame(flat_data)
df.head()

# COMMAND ----------

 dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=lambda x: list(zip(*x))
    )

# COMMAND ----------

df

# COMMAND ----------


