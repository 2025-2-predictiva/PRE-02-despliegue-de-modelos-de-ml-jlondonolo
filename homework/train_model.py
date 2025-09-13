import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Cargar los datos
file_path = './files/input/house_data.csv'
df = pd.read_csv(file_path)

# Seleccionar características y variable objetivo
features = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "condition"]
X = df[features]
y = df["price"]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"MSE: {mse:.2f}")

# Guardar el modelo entrenado
joblib.dump(model, './homework/house_predictor.pkl')
print("Modelo guardado como house_predictor.pkl")
