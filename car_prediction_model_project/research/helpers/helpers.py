import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, auc


def df_info(dataframe):
    columns_info = pd.DataFrame({"nunique" : dataframe.nunique().values,
                                 "isnull" : dataframe.isnull().sum().values,
                                 "type" : dataframe.dtypes.values}, 
                                 index = dataframe.nunique().index)
    print(columns_info)

def iqr_values(dataframe, col, upper_quantile=0.75, lower_quantile=0.25, info=False, replace_with_limit=False):
    IQR = dataframe[col].quantile(0.75) - dataframe[col].quantile(0.25)
    lower_limit = dataframe[col].quantile(lower_quantile) - IQR*1.5
    upper_limit = dataframe[col].quantile(upper_quantile) + IQR*1.5
    under_lower = dataframe[dataframe[col] < lower_limit].shape[0]
    over_upper = dataframe[dataframe[col] > upper_limit].shape[0]
    if replace_with_limit:
        for index, val in enumerate(dataframe[col]):
            if val > upper_limit:
                dataframe[col].iloc[index] = upper_limit
            elif val < lower_limit:
                dataframe[col].iloc[index] = lower_limit
    if info:
        print(f"Low Limit : {lower_limit}\n",
              f"Up Limit : {upper_limit}\n",
              f"{under_lower} observation units under 'Low Limit'\n",
              f"{over_upper} observation units over 'Up Limit'")
    return lower_limit, upper_limit, under_lower, over_upper

def plot_importance(model, original_columns, feature_num=20,save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': original_columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:feature_num])
    plt.title('Features Importances')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

def error_metrics(x_train_or_test, y_train_or_test, title, model_type, final_model=None, metric="rmse", plot_graph=False, figure_height=10, figure_size=10):
    for i in range(len(x_train_or_test)):
        y_pred = final_model.predict(x_train_or_test[i])
        mae = mean_absolute_error(y_train_or_test[i], y_pred)
        mse = mean_squared_error(y_train_or_test[i], y_pred)
        rmse = mean_squared_error(y_train_or_test[i], y_pred, squared=False)
        r2 = r2_score(y_train_or_test[i], y_pred)

        y_true_log = np.log1p(y_train_or_test[i])
        y_pred_log = np.log1p(y_pred)
        rmsle = np.sqrt(mean_squared_error(y_true_log, y_pred_log))

        print(f"\n\n{title[i]} Error:\n")
        print("Mean Absolute Error (MAE):", mae)
        print("Mean Squared Error (MSE):", mse)
        print("Root Mean Squared Error (RMSE):", rmse)
        print("R^2 Score:", r2)
        print("RMSLE : ", rmsle)
    
    if plot_graph:
        if model_type == 'LGBM':            
            train_scores = final_model.evals_result_['training'][metric]
            test_scores = final_model.evals_result_['valid_1'][metric]

            train_area = auc(range(len(train_scores)), train_scores)
            test_area = auc(range(len(test_scores)), test_scores)
            area_between_curves = abs(train_area - test_area)

            print("\n\nAreas under & between curves: {:.2f}".format(train_area))
            print("\nArea Under Train set:")
            print("Area Under Test set: {:.2f}".format(test_area))
            print("Area Between curves: {:.2f}".format(area_between_curves))
            print("\n\nError Graphic for per iter:")

            plt.figure(figsize=(figure_size, figure_height))
            plt.plot(train_scores, label='Train')
            plt.plot(test_scores, label='Test')
            plt.fill_between(range(len(train_scores)), train_scores, test_scores, alpha=0.1, color='orange')
            plt.text(len(train_scores)//2, (max(train_scores)+min(test_scores))/2, 
                     "Area Between curves: {:.2f}".format(area_between_curves), ha="center", va="center")
            plt.xlabel('Iteration')
            plt.ylabel('RMSE')
            plt.title('LGBM Model')
            plt.legend()
            plt.show()
        
        elif model_type == 'XGBOOST':
            train_scores = final_model.evals_result()['validation_0'][metric]
            test_scores = final_model.evals_result()['validation_1'][metric]

            train_area = auc(range(len(train_scores)), train_scores)
            test_area = auc(range(len(test_scores)), test_scores)
            area_between_curves = abs(train_area - test_area)

            print("\n\nAreas under & between curves: {:.2f}".format(train_area))
            print("\nArea Under Train set:")
            print("Area Under Test set: {:.2f}".format(test_area))
            print("Area Between curves: {:.2f}".format(area_between_curves))
            print("\n\nError Graphic for per iter:")

            plt.figure(figsize=(figure_size, figure_height))
            plt.plot(range(1, len(train_scores)+1), train_scores, label='Train')
            plt.plot(range(1, len(test_scores)+1), test_scores, label='Test')
            plt.fill_between(range(len(train_scores)), train_scores, test_scores, alpha=0.1, color='orange')
            plt.text(len(train_scores)//2, (max(train_scores)+min(test_scores))/2, 
                     "Area Between curves: {:.2f}".format(area_between_curves), ha="center", va="center")
            plt.xlabel('n_estimators')
            plt.ylabel('RMSE')
            plt.title('XGBoost Model')
            plt.legend()
            plt.show()
        else:
            print("Please enter your 'model_type'!!!")