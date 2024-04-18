import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
)
from tqdm.notebook import tqdm

from utils import (
    prepare_data,
    create_dmatrix,
    reshape_y_from_dmatrix,
    get_confidence_interval,
    provide_stratified_bootstap_sample_indices,
    produce_stats,
)


FRACTURES = ["vertebral_fracture", "hip_fracture", "any_fracture"]

# Define times at which to measure AUC (every year)
TIMES = np.arange(12, 95, 12)
TIME_TO_EVALUATE = 24


def train_model(
    train_data,
    best_params,
    t,
    times,
    model_type,
    calibrated=False,
    cv=5,
    fractures=FRACTURES,
    features=[],
    only_first_visits=False,
):
    # Setup dataframe to store statistics in
    multi_index = pd.MultiIndex.from_tuples(
        [
            (x, y)
            for x in ["vertebral", "hip", "any"]
            for y in ["train", "valid", "test"]
        ]
    )
    df_stats = pd.DataFrame(
        [],
        index=multi_index,
        columns=["harrel_global", "harrel_2y", "uno_2y", "auc_2y", "auc_mean"],
    )

    kf = StratifiedKFold(n_splits=cv)
    _, ax = plt.subplots(1, 1, figsize=(10, 7))
    for i, fx in enumerate(fractures):
        fx_type = fx.split("_")[0]
        print(fx_type)
        params = best_params[fx_type]
        n_trees = params.pop("num_boost_round")

        c_index_scores = {"train": {}, "valid": {}}
        auc_scores = {}
        for train_index, validation_index in kf.split(
            train_data, train_data.loc[:, "any_right_censored"]
        ):
            # Split train and validation set
            train_data_prep = prepare_data(
                train_data.loc[train_index, :],
                fx_type,
                only_first_visits=only_first_visits,
                model_type=model_type,
                feature_list=features,
            )
            valid_data_prep = prepare_data(
                train_data.loc[validation_index, :],
                fx_type,
                only_first_visits=only_first_visits,
                model_type=model_type,
                feature_list=features,
            )

            # Create DMatrix for training and validation
            dtrain = create_dmatrix(train_data_prep, model_type)
            dvalid = create_dmatrix(valid_data_prep, model_type)

            # Train gradient boosted trees using AFT loss and metric
            bst = xgb.train(params, dtrain, num_boost_round=n_trees, verbose_eval=False)

            # Predict time-to-event on train and validation set
            train_predicted = bst.predict(dtrain)
            valid_predicted = bst.predict(dvalid)

            if calibrated:
                # Train COX PH model on validation set
                cox_model = CoxPHSurvivalAnalysis()
                y_valid = reshape_y_from_dmatrix(dvalid, model_type)
                cox_model.fit(
                    valid_predicted.reshape(-1, 1), y_valid
                )  # Reshape to 2D array (only one feature)

                # Predict cumulative hazard functions of train and validation set
                chf_funcs_train = cox_model.predict_cumulative_hazard_function(
                    train_predicted.reshape(-1, 1), return_array=False
                )
                chf_funcs_valid = cox_model.predict_cumulative_hazard_function(
                    valid_predicted.reshape(-1, 1), return_array=False
                )

            elif model_type == "aft":
                # Limit predicted output to 100'000 (larger values can lead to computation errors)
                max_value = 100000
                train_predicted = np.where(
                    train_predicted < max_value, train_predicted, max_value
                )
                valid_predicted = np.where(
                    valid_predicted < max_value, valid_predicted, max_value
                )
                # we take the negative predicted values
                # since c_index_ipcw expects risks instead of time-to-event values
                train_predicted = -train_predicted
                valid_predicted = -valid_predicted

            # Define dictionary for c-index calculation
            y_dict = {
                "train": {
                    "y": reshape_y_from_dmatrix(dtrain, model_type),
                    "y_censored": reshape_y_from_dmatrix(
                        dtrain, model_type, censored_after=t
                    ),
                    "y_pred": [chf(t) for chf in chf_funcs_train]
                    if calibrated
                    else train_predicted,
                },
                "valid": {
                    "y": reshape_y_from_dmatrix(dvalid, model_type),
                    "y_censored": reshape_y_from_dmatrix(
                        dvalid, model_type, censored_after=t
                    ),
                    "y_pred": [chf(t) for chf in chf_funcs_valid]
                    if calibrated
                    else valid_predicted,
                },
            }

            for tv in ["train", "valid"]:
                y = y_dict[tv]["y"]
                y_censored = y_dict[tv]["y_censored"]
                y_pred = y_dict[tv]["y_pred"]

                # Compute C-Index (harrel and uno)
                c_index_harrel_glob, _, _, _, _ = concordance_index_censored(
                    y["Event_observed"], y["Survival_in_months"], y_pred
                )
                c_index_harrel, _, _, _, _ = concordance_index_censored(
                    y_censored["Event_observed"],
                    y_censored["Survival_in_months"],
                    y_pred,
                )
                c_index_uno, _, _, _, _ = concordance_index_ipcw(
                    y_dict["train"]["y"], y_censored, y_pred
                )

                c_index_scores[tv].setdefault(
                    f"{fx_type}_harrel_global", list()
                ).append(c_index_harrel_glob)
                c_index_scores[tv].setdefault(f"{fx_type}_harrel", list()).append(
                    c_index_harrel
                )
                c_index_scores[tv].setdefault(f"{fx_type}_uno", list()).append(
                    c_index_uno
                )

            # Compute cumulative/dynamic AUC
            if calibrated:
                risk_scores = np.row_stack([chf(times) for chf in chf_funcs_valid])
            else:
                risk_scores = np.broadcast_to(
                    valid_predicted, (times.shape[0], valid_predicted.shape[0])
                ).T
            auc, mean_auc = cumulative_dynamic_auc(
                y_dict["train"]["y"], y_dict["valid"]["y"], risk_scores, times
            )
            auc_scores.setdefault(fx_type, list()).append(auc)
            auc_scores.setdefault(f"{fx_type}_mean", list()).append(mean_auc)

        # Print C-Index Scores
        df_stats = produce_stats(df_stats, c_index_scores, auc_scores, t=t)

        # Plot the cumulative/dynamic AUC scores
        ax.plot(
            times,
            np.mean(auc_scores[fx_type], axis=0),
            "o-",
            label="{} (mean AUC = {:.3f})".format(
                fx_type, np.mean(auc_scores[f"{fx_type}_mean"])
            ),
        )

        # calculate error bar boundaries (+/- standard deviation)
        yerr_neg = np.mean(auc_scores[fx_type], axis=0) - np.std(
            auc_scores[fx_type], axis=0
        )
        yerr_pos = np.mean(auc_scores[fx_type], axis=0) + np.std(
            auc_scores[fx_type], axis=0
        )
        ax.fill_between(times, yerr_neg, yerr_pos, color=f"C{i}", alpha=0.25)

    ax.set_title("Validation cumulative/dynamic AUC")
    ax.set_ylim(0.5, 0.85)
    ax.set_xlabel("months from first visit")
    ax.set_ylabel("time-dependent AUC (5-fold CV)")
    ax.legend(loc="upper center")
    ax.grid(True)
    print()

    return df_stats


def test_model(
    train_data,
    test_data,
    best_params,
    t,
    times,
    modelname=None,
    df_stats=pd.DataFrame(),
    calibrated=False,
    fractures=FRACTURES,
    save_model_path=None,
    features=[],
    save_results_path=None,
    only_first_visits=False,
    bootstrap=True,
    plot=True,
):
    random.seed(99)
    model_type = modelname.split("_")[0]

    if plot:
        _, ax = plt.subplots(1, 1, figsize=(10, 7))
    test_scores = {}
    for i, fx in enumerate(fractures):
        fx_type = fx.split("_")[0]
        print(fx_type)

        # Use full training data
        train_data_prep = prepare_data(
            train_data,
            fx_type,
            only_first_visits=only_first_visits,
            model_type=model_type,
            feature_list=features,
        )
        # Create D-Matrices used for training
        dtrain = create_dmatrix(train_data_prep, model_type)

        # Train gradient boosted trees using AFT loss and metric
        params = best_params[fx_type]
        n_trees = params.pop("num_boost_round")
        bst = xgb.train(params, dtrain, num_boost_round=n_trees, verbose_eval=False)

        if calibrated:
            train_predicted = bst.predict(dtrain)
            # Train COX PH model on validation set
            cox_model = CoxPHSurvivalAnalysis()
            y_train = reshape_y_from_dmatrix(dtrain, model_type)

            # Reshape to 2D array (only one feature)
            cox_model.fit(train_predicted.reshape(-1, 1), y_train)

        if save_model_path:
            path = f"{save_model_path}/xgb_{modelname}"
            # Create directory if not exists
            if not os.path.exists(path):
                os.mkdir(path)
                print("Created directory: ", path)
            bst.save_model(f"{path}/{fx_type}.json")

        c_index_scores = {"harrel_global": [], "harrel_2y": [], "uno_2y": []}
        auc_scores = {"auc": [], "mean_auc": []}
        # If bootstrap, subsample test set and evaluate on 5 different folds
        if bootstrap:
            # Bootstrap test set with 100 samples
            n_bootstrap = 1000
            for _ in tqdm(range(n_bootstrap)):
                bs_sample = test_data.copy()

                # Create bootstrap sample
                # random.seed(bootsrap_idx)
                bs_index_list_stratified = provide_stratified_bootstap_sample_indices(
                    bs_sample, fx_type
                )
                bs_sample = bs_sample.loc[bs_index_list_stratified, :].reset_index(
                    drop=True
                )
                test_data_prep = prepare_data(
                    bs_sample,
                    fx_type,
                    only_first_visits=True,
                    model_type=model_type,
                    feature_list=features,
                )
                # Create D-Matrices used for testing
                dtest = create_dmatrix(test_data_prep, model_type)

                # Predict time-to-event on train and validation set
                test_predicted = bst.predict(dtest)

                if calibrated:
                    # Predict cumulative hazard functions
                    chf_funcs_test = cox_model.predict_cumulative_hazard_function(
                        test_predicted.reshape(-1, 1)
                    )
                    risk_scores = np.row_stack([chf(times) for chf in chf_funcs_test])
                    y_pred = [chf(t) for chf in chf_funcs_test]

                else:
                    if model_type == "aft":
                        # Limit predicted output to 100'000 (larger values can lead to computation errors)
                        max_value = 100000
                        test_predicted = np.where(
                            test_predicted < max_value, test_predicted, max_value
                        )
                        test_predicted = -test_predicted
                    risk_scores = np.broadcast_to(
                        test_predicted, (times.shape[0], test_predicted.shape[0])
                    ).T
                    y_pred = test_predicted

                # Reshape y for c-index calculation with
                y_train = reshape_y_from_dmatrix(dtrain, model_type)
                y_test = reshape_y_from_dmatrix(dtest, model_type)
                y_test_censored = reshape_y_from_dmatrix(
                    dtest, model_type, censored_after=t
                )

                # Compute scores
                c_index_harrel_glob, _, _, _, _ = concordance_index_censored(
                    y_test["Event_observed"], y_test["Survival_in_months"], y_pred
                )
                c_index_harrel, _, _, _, _ = concordance_index_censored(
                    y_test_censored["Event_observed"],
                    y_test_censored["Survival_in_months"],
                    y_pred,
                )
                c_index_uno, _, _, _, _ = concordance_index_ipcw(
                    y_train, y_test_censored, y_pred
                )
                c_index_scores["harrel_global"].append(c_index_harrel_glob)
                c_index_scores["harrel_2y"].append(c_index_harrel)
                c_index_scores["uno_2y"].append(c_index_uno)

                # Compute cumulative/dynamic AUC
                auc, mean_auc = cumulative_dynamic_auc(
                    y_train, y_test, risk_scores, times
                )
                auc_scores["auc"].append(auc)
                auc_scores["mean_auc"].append(mean_auc)

            # Save c-index scores of each bootstrap round
            if save_results_path:
                results_path = f"{save_results_path}/c_index_scores/xgb/{modelname}{'_calibrated' if calibrated else ''}_{fx_type}.json"
                with open(results_path, "w") as f:
                    json.dump(c_index_scores, f)

            # 95% confidence intervals
            harrel_lower, harrel_upper = get_confidence_interval(
                c_index_scores["harrel_global"], alpha=0.95, decimals=3
            )
            uno_lower, uno_upper = get_confidence_interval(
                c_index_scores["uno_2y"], alpha=0.95, decimals=3
            )
            test_scores[fx_type] = {
                "harrel_global": {
                    "c-index": round(np.mean(c_index_scores["harrel_global"]), 4),
                    "lower": harrel_lower,
                    "upper": harrel_upper,
                },
                "harrel_2y": round(np.mean(c_index_scores["harrel_2y"]), 4),
                "uno_2y": {
                    "c-index": round(np.mean(c_index_scores["uno_2y"]), 4),
                    "lower": uno_lower,
                    "upper": uno_upper,
                },
                "auc": round(
                    np.mean(auc_scores["auc"], axis=0)[(TIME_TO_EVALUATE // 12) - 1], 4
                ),
                "auc_mean": round(np.mean(auc_scores["mean_auc"]), 4),
            }
            print(test_scores[fx_type])

            # Compute and plot cululative/dynamic AUC
            ax.plot(
                times,
                np.mean(auc_scores["auc"], axis=0),
                "o-",
                color=f"C{i}",
                label="{} (mean AUC = {:.3f})".format(
                    fx_type, np.mean(auc_scores["mean_auc"])
                ),
            )

            # 95% confidence intervals
            lower, upper = get_confidence_interval(auc_scores["auc"], alpha=0.95, ax=0)
            ax.fill_between(times, lower, upper, color=f"C{i}", alpha=0.25)

            ax.set_title("Test cumulative/dynamic AUC")
            ax.set_ylim(0.5, 0.85)
            ax.set_xlabel("months from first visit")
            ax.set_ylabel("time-dependent AUC")
            ax.legend(loc="upper center")
            ax.grid(True)

            # Add to df_stats
            df_stats.loc[(fx_type, "test"), "harrel_global"] = test_scores[fx_type][
                "harrel_global"
            ]["c-index"]
            df_stats.loc[(fx_type, "test"), "harrel_2y"] = test_scores[fx_type][
                "harrel_2y"
            ]
            df_stats.loc[(fx_type, "test"), "uno_2y"] = test_scores[fx_type]["uno_2y"][
                "c-index"
            ]
            df_stats.loc[(fx_type, "test"), "auc_2y"] = test_scores[fx_type]["auc"]
            df_stats.loc[(fx_type, "test"), "auc_mean"] = test_scores[fx_type][
                "auc_mean"
            ]

            print()
        else:
            test_data_prep = prepare_data(
                test_data,
                fx_type,
                only_first_visits=True,
                model_type=model_type,
                feature_list=features,
            )
            # Create D-Matrices used for testing
            dtest = create_dmatrix(test_data_prep, model_type)

            # Predict time-to-event on train and validation set
            test_predicted = bst.predict(dtest)

            if calibrated:
                # Predict cumulative hazard functions
                chf_funcs_test = cox_model.predict_cumulative_hazard_function(
                    test_predicted.reshape(-1, 1)
                )
                risk_scores = np.row_stack([chf(times) for chf in chf_funcs_test])
                y_pred = [chf(t) for chf in chf_funcs_test]

            else:
                if model_type == "aft":
                    # Limit predicted output to 100'000 (larger values can lead to computation errors)
                    max_value = 100000
                    test_predicted = np.where(
                        test_predicted < max_value, test_predicted, max_value
                    )
                    test_predicted = -test_predicted
                risk_scores = np.broadcast_to(
                    test_predicted, (times.shape[0], test_predicted.shape[0])
                ).T
                y_pred = test_predicted

            # Reshape y for c-index calculation with
            y_train = reshape_y_from_dmatrix(dtrain, model_type)
            y_test = reshape_y_from_dmatrix(dtest, model_type)
            y_test_censored = reshape_y_from_dmatrix(
                dtest, model_type, censored_after=t
            )

            # Compute scores
            c_index_harrel_glob, _, _, _, _ = concordance_index_censored(
                y_test["Event_observed"], y_test["Survival_in_months"], y_pred
            )
            c_index_harrel, _, _, _, _ = concordance_index_censored(
                y_test_censored["Event_observed"],
                y_test_censored["Survival_in_months"],
                y_pred,
            )
            c_index_uno, _, _, _, _ = concordance_index_ipcw(
                y_train, y_test_censored, y_pred
            )

            print("Harrel C-Index (global):", round(c_index_harrel_glob, 4))
            print("Harrel C-Index (t=2y):", round(c_index_harrel, 4))
            print("Uno C-Index (t=2y):", round(c_index_uno, 4))

    # Save test scores
    if save_results_path:
        with open(f"{save_results_path}/xgb_{modelname}_test_scores.json", "w") as f:
            json.dump(test_scores, f)

    summary = {"stats": df_stats, "xgb": bst}
    if calibrated:
        summary["cox"] = cox_model
    return summary
