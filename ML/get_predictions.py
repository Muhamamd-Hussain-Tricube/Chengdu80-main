from tensorflow.keras.models import load_model

model_open_cnn = load_model("model_open_cnn.h5")
model_high_cnn = load_model("model_high_cnn.h5")
model_low_cnn = load_model("model_low_cnn.h5")
model_close_cnn = load_model("model_close_cnn.h5")

model_open_lstm = load_model("model_open_lstm.h5")
model_high_lstm = load_model("model_high_lstm.h5")
model_low_lstm = load_model("model_low_lstm.h5")
model_close_lstm = load_model("model_close_lstm.h5")

model_open_rf = load_model("model_open_rf.h5")
model_high_rf = load_model("model_high_rf.h5")
model_low_rf = load_model("model_low_rf.h5")
model_close_rf = load_model("model_close_rf.h5")

model_open_svm = load_model("model_open_svm.h5")
model_high_svm = load_model("model_high_svm.h5")
model_low_svm = load_model("model_low_svm.h5")
model_close_svm = load_model("model_close_svm.h5")




# Next day prediction
# Make predictions for the 'Open' price
pred_open_cnn_next_day = model_open_cnn.predict(X_test_cnn[-1:]).flatten()
pred_open_lstm_next_day = model_open_lstm.predict(X_test_lstm[-1:]).flatten()
pred_open_rf_next_day = model_open_rf.predict(X_test[-1:]).flatten()
pred_open_svm_next_day = model_open_svm.predict(X_test[-1:]).flatten()

# Create a DataFrame for the next day's data for the 'Open' price
X_next_day_xgb_open = pd.DataFrame({'Open_cnn': pred_open_cnn_next_day, 'Open_lstm': pred_open_lstm_next_day, 'Open_rf': pred_open_rf_next_day, 'Open_svm': pred_open_svm_next_day})

# Make predictions for the 'Open' price using the XGBoost model
pred_open_xgb_next_day = xgb_open.predict(X_next_day_xgb_open)


# Make predictions for the 'HIGH' price
pred_high_cnn_next_day = model_high_cnn.predict(X_test_cnn[-1:]).flatten()
pred_high_lstm_next_day = model_high_lstm.predict(X_test_lstm[-1:]).flatten()
pred_high_rf_next_day = model_high_rf.predict(X_test[-1:]).flatten()
pred_high_svm_next_day = model_high_svm.predict(X_test[-1:]).flatten()

# Create a DataFrame for the next day's data for the 'high' price
X_next_day_xgb_high = pd.DataFrame({'High_cnn': pred_high_cnn_next_day, 'High_lstm': pred_high_lstm_next_day, 'High_rf': pred_high_rf_next_day, 'High_svm': pred_high_svm_next_day})

# Make predictions for the 'high' price using the XGBoost model
pred_high_xgb_next_day = xgb_high.predict(X_next_day_xgb_high)

# Make predictions for the 'low' price
pred_low_cnn_next_day = model_low_cnn.predict(X_test_cnn[-1:]).flatten()
pred_low_lstm_next_day = model_low_lstm.predict(X_test_lstm[-1:]).flatten()
pred_low_rf_next_day = model_low_rf.predict(X_test[-1:]).flatten()
pred_low_svm_next_day = model_low_svm.predict(X_test[-1:]).flatten()

# Create a DataFrame for the next day's data for the 'Open' price
X_next_day_xgb_low = pd.DataFrame({'Low_cnn': pred_low_cnn_next_day, 'Low_lstm': pred_low_lstm_next_day, 'Low_rf': pred_low_rf_next_day, 'Low_svm': pred_low_svm_next_day})

# Make predictions for the 'Low' price using the XGBoost model
pred_low_xgb_next_day = xgb_low.predict(X_next_day_xgb_low)


# Make predictions for the 'Close' price
pred_close_cnn_next_day = model_close_cnn.predict(X_test_cnn[-1:]).flatten()
pred_close_lstm_next_day = model_close_lstm.predict(X_test_lstm[-1:]).flatten()
pred_close_rf_next_day = model_close_rf.predict(X_test[-1:]).flatten()
pred_close_svm_next_day = model_close_svm.predict(X_test[-1:]).flatten()

# Create a DataFrame for the next day's data for the 'Close' price
X_next_day_xgb_close = pd.DataFrame({'Close_cnn': pred_close_cnn_next_day, 'Close_lstm': pred_close_lstm_next_day, 'Close_rf': pred_close_rf_next_day, 'Close_svm': pred_close_svm_next_day})

# Make predictions for the 'Close' price using the XGBoost model
pred_close_xgb_next_day = xgb_close.predict(X_next_day_xgb_close)

# Store the predictions for the next day's 'Close' price
next_day_predictions = pd.DataFrame({'Open': pred_open_xgb_next_day, 'High':pred_high_xgb_next_day,'Low':pred_low_xgb_next_day,'Close':pred_close_xgb_next_day})

print(next_day_predictions)