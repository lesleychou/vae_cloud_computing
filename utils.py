from datetime import datetime
from lzma import CHECK_CRC32
import numpy as np
import torch
from rdt.transformers import numerical, categorical
import pandas as pd

# Graph Visualisation
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 54
plt.rcParams["figure.figsize"] = (5,9)

from sklearn.preprocessing import StandardScaler, RobustScaler


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# -------- Pre-Processing for MIMIC sets -------- #
# Internal sets provided by NHSX - outside users will have to stick with SUPPORT set


def mimic_pre_proc(data_supp, original_continuous_columns, original_categorical_columns, pre_proc_method="GMM"):
    # Specify column configurations
    # original_categorical_columns = [
    #     'load-interval'
    # ]
    # original_continuous_columns = ['output-packet-rate']

    categorical_columns = original_categorical_columns.copy()
    continuous_columns = original_continuous_columns.copy()

    # As of coding this, new version of RDT adds in GMM transformer which is what we require, however hyper transformers do not work as individual
    # transformers take a 'columns' argument that can only allow for fitting of one column - so you need to loop over and create one for each column
    # in order to fit the dataset - https://github.com/sdv-dev/RDT/issues/376

    continuous_transformers = {}
    categorical_transformers = {}

    transformed_dataset = data_supp.copy(deep=True)

    # Define columns based on datatype and then loop over creating and fitting
    # transformers

    # WE NEED TO RETAIN THIS SET AS METRICS DO NOT EVALUATE WITH DATETIMES BUT THEY WILL EVALUATE
    # IF DATETIMES ARE IN A SECONDS FORMAT

    original_metric_set = transformed_dataset.copy(deep=True)

    if pre_proc_method == "GMM":

        for index, column in enumerate(continuous_columns):

            # Fit GMM
            temp_continuous = numerical.BayesGMMTransformer(
                # random_state=gmm_seed
            )
            temp_continuous.fit(transformed_dataset, columns=column)
            continuous_transformers[
                "continuous_{}".format(column)
            ] = temp_continuous

            categorical_columns += [str(column) + ".component"]

            transformed_dataset = temp_continuous.transform(
                transformed_dataset
            )

        # Each numerical one gets a .normalized column + a .component column giving the mixture info
        # This too needs to be one hot encoded

        continuous_columns = [
            str(col) + ".normalized" for col in continuous_columns
        ]
    elif pre_proc_method == "standard":

        for index, column in enumerate(continuous_columns):

            # Fit sklearn standard scaler to each column
            temp_continuous = StandardScaler()
            temp_column = transformed_dataset[column].values.reshape(-1, 1)
            temp_continuous.fit(temp_column)
            continuous_transformers[
                "continuous_{}".format(column)
            ] = temp_continuous

            transformed_dataset[column] = (
                temp_continuous.transform(temp_column)
            ).flatten()

    elif pre_proc_method == "robust":
        for index, column in enumerate(continuous_columns):
            # Fit RobustScaler to each column
            temp_continuous = RobustScaler()
            temp_column = transformed_dataset[column].values.reshape(-1, 1)
            temp_continuous.fit(temp_column)
            continuous_transformers["continuous_{}".format(column)] = temp_continuous

            transformed_dataset[column] = (
                temp_continuous.transform(temp_column)
            ).flatten()

    num_categories = []

    for col in categorical_columns:
        if transformed_dataset[col].dtype == float:
            # Convert to int
            transformed_dataset[col] = transformed_dataset[col].astype(int)

        if transformed_dataset[col].dtype == int:
            num_of_diff_catagories = len(transformed_dataset[col].unique())
            num_categories.append(num_of_diff_catagories)

        # Categories are strings/objects
        else:
            # Convert column into one type
            values = np.unique(
                transformed_dataset[col].astype(str), return_counts=False
            )

            num_categories.append(values.shape[0])

    num_continuous = len(continuous_columns)

    for index, column in enumerate(categorical_columns):
        print(index, column, "------index, column")
        temp_categorical = categorical.OneHotEncodingTransformer()
        temp_categorical.fit(transformed_dataset, columns=column)
        categorical_transformers[
            "categorical_{}".format(index)
        ] = temp_categorical

        transformed_dataset = temp_categorical.transform(transformed_dataset)

    # We need the dataframe in the correct format i.e. categorical variables first and in the order of
    # num_categories with continuous variables placed after

    reordered_dataframe = transformed_dataset.iloc[:, num_continuous:]

    reordered_dataframe = pd.concat(
        [reordered_dataframe, transformed_dataset.iloc[:, :num_continuous]],
        axis=1,
    )

    x_train_df = reordered_dataframe.to_numpy()
    x_train = x_train_df.astype("float32")

    return (
        x_train,
        original_metric_set,
        reordered_dataframe.columns,
        continuous_transformers,
        categorical_transformers,
        num_categories,
        num_continuous,
    )


# -------- Reverse Transformations -------- #


def reverse_transformers(
    synthetic_set,
    data_supp_columns,
    cont_transformers=None,
    cat_transformers=None,
    date_transformers=None,
    pre_proc_method="GMM",
):

    # Now all of the transformations from the dictionary - first loop over the categorical columns

    synthetic_transformed_set = synthetic_set.copy(deep=True)

    if cat_transformers != None:
        for transformer_name in cat_transformers:

            transformer = cat_transformers[transformer_name]
            column_name = transformer_name[12:]

            synthetic_transformed_set = transformer.reverse_transform(
                synthetic_transformed_set
            )

    if cont_transformers != None:

        if pre_proc_method == "GMM":

            for transformer_name in cont_transformers:

                transformer = cont_transformers[transformer_name]
                column_name = transformer_name[11:]

                synthetic_transformed_set = transformer.reverse_transform(
                    synthetic_transformed_set
                )

        elif pre_proc_method == "standard":

            for transformer_name in cont_transformers:

                transformer = cont_transformers[transformer_name]
                column_name = transformer_name[11:]

                # Reverse the standard scaling
                synthetic_transformed_set[
                    column_name
                ] = transformer.inverse_transform(
                    synthetic_transformed_set[column_name].values.reshape(
                        -1, 1
                    )
                ).flatten()
        # reverse the robust scaling
        elif pre_proc_method == "robust":
            for transformer_name in cont_transformers:

                transformer = cont_transformers[transformer_name]
                column_name = transformer_name[11:]

                # Reverse the robust scaling
                synthetic_transformed_set[
                    column_name
                ] = transformer.inverse_transform(
                    synthetic_transformed_set[column_name].values.reshape(
                        -1, 1
                    )
                ).flatten()


    if date_transformers != None:
        for transformer_name in date_transformers:

            transformer = date_transformers[transformer_name]
            column_name = transformer_name[9:]

            synthetic_transformed_set = transformer.reverse_transform(
                synthetic_transformed_set
            )

    synthetic_transformed_set = pd.DataFrame(
        synthetic_transformed_set, columns=data_supp_columns
    )

    return synthetic_transformed_set


# -------- Constraint based sampling for MIMIC work -------- #


def constraint_filtering(
    n_rows,
    vae,
    reordered_cols,
    data_supp_columns,
    cont_transformers,
    cat_transformers,
    date_transformers,
    reverse_transformers=reverse_transformers,
    pre_proc_method="GMM",
):

    # Generate samples
    synthetic_trial = vae.generate(n_rows)

    if torch.cuda.is_available():
        # Create pandas dataframe in column order
        synthetic_dataframe = pd.DataFrame(
            synthetic_trial.cpu().detach().numpy(), columns=reordered_cols
        )
    else:
        # Create pandas dataframe in column order
        synthetic_dataframe = pd.DataFrame(
            synthetic_trial.detach().numpy(), columns=reordered_cols
        )

    # Reverse all the transformations ready for filtering
    synthetic_dataframe = reverse_transformers(
        synthetic_dataframe,
        data_supp_columns,
        cont_transformers,
        cat_transformers,
        date_transformers,
        pre_proc_method=pre_proc_method,
    )

    # Function to filter out the constraints from the set - returns valid dataframe
    def constraint_check(synthetic_df):
        # age greater than 0                   patient was discharged after being admitted           patient admitted after their date of birth         patient first chart after admit time
        valid_df = synthetic_df[
            (synthetic_df["age"] > 0)
            | (synthetic_df["DISCHTIME"] >= synthetic_df["ADMITTIME"])
            | (synthetic_df["ADMITTIME"] > synthetic_df["DOB"])
            | (synthetic_df["CHARTTIME"] >= synthetic_df["ADMITTIME"])
        ]

        return valid_df

    # Do first check
    synthetic_dataframe = constraint_check(synthetic_dataframe)

    # Loop over returning a valid dataframe each time until we get a set that is big enough
    while synthetic_dataframe.shape[0] != n_rows:

        rows_needed = n_rows - synthetic_dataframe.shape[0]

        # If we have too many, remove the required amount
        if rows_needed < 0:

            rows_needed = np.arange(abs(rows_needed))

            # Drop the bottom rows_needed amount

            synthetic_dataframe.drop(rows_needed, axis=0, inplace=True)

        # Need to generate enough to fill the dataframe
        else:

            new_set = vae.generate(rows_needed)

            new_set = pd.DataFrame(
                new_set.cpu().detach().numpy(), columns=reordered_cols
            )

            new_set = reverse_transformers(
                new_set,
                data_supp_columns,
                cont_transformers,
                cat_transformers,
                date_transformers,
            )

            new_filtered_set = constraint_check(new_set)

            # Add this onto the original and re-run
            synthetic_dataframe = pd.concat(
                [synthetic_dataframe, new_filtered_set]
            )

    return synthetic_dataframe


def plot_elbo(
    n_epochs,
    log_elbo,
    log_reconstruction,
    log_divergence,
    saving_filepath=None,
    pre_proc_method="GMM",
):

    x = np.arange(n_epochs)

    y1 = log_elbo
    y2 = log_reconstruction
    y3 = log_divergence

    plt.plot(x, y1, label="ELBO")
    plt.plot(x, y2, label="RECONSTRUCTION")
    plt.plot(x, y3, label="DIVERGENCE")
    plt.xlabel("Number of Epochs")
    # Set the y axis label of the current axis.
    plt.ylabel("Loss Value")
    # Set a title of the current axes.
    plt.title("ELBO Breakdown")
    # show a legend on the plot
    plt.legend()

    if saving_filepath != None:
        # Save static image
        plt.savefig(
            "{}ELBO_Breakdown_SynthVAE_{}.png".format(
                saving_filepath, pre_proc_method
            )
        )

    plt.show()

    return None


def plot_likelihood_breakdown(
    n_epochs,
    log_categorical,
    log_numerical,
    saving_filepath=None,
    pre_proc_method="GMM",
):

    x = np.arange(n_epochs)

    y1 = log_categorical
    y2 = log_numerical

    plt.subplot(1, 2, 1)
    plt.plot(x, y1, label="CATEGORICAL")
    plt.xlabel("Number of Epochs")
    # Set the y axis label of the current axis.
    plt.ylabel("Loss Value")
    # Set a title of the current axes.
    plt.title("Categorical Breakdown")
    # show a legend on the plot
    plt.subplot(1, 2, 2)
    plt.plot(x, y2, label="NUMERICAL")
    plt.xlabel("Number of Epochs")
    # Set the y axis label of the current axis.
    plt.ylabel("Loss Value")
    # Set a title of the current axes.
    plt.title("Numerical Breakdown")
    # show a legend on the plot
    plt.tight_layout()

    if saving_filepath != None:
        # Save static image
        plt.savefig(
            "{}Reconstruction_Breakdown_SynthVAE_{}.png".format(
                saving_filepath, pre_proc_method
            )
        )

    return None


def plot_variable_distributions(
    categorical_columns,
    continuous_columns,
    data_supp,
    synthetic_supp,
    saving_filepath=None,
    pre_proc_method="GMM",
):

    # Plot some examples using plotly

    for column in categorical_columns:

        plt.subplot(1, 2, 1)
        plt.hist(x=synthetic_supp[column])
        plt.title("Synthetic")
        # Set the x axis label of the current axis
        plt.xlabel("Data Value")
        # Set the y axis label of the current axis.
        plt.ylabel("Distribution")
        # Set a title of the current axes.
        plt.title("Synthetic".format(column))
        # show a legend on the plot
        plt.subplot(1, 2, 2)
        plt.hist(x=data_supp[column])
        plt.title("Original")
        # Set the x axis label of the current axis
        plt.xlabel("Data Value")
        # Set the y axis label of the current axis.
        plt.ylabel("Distribution")
        # Set a title of the current axes.
        plt.title("Original".format(column))
        # show a legend on the plot
        plt.suptitle("Variable {}".format(column))

        plt.tight_layout()

        if saving_filepath != None:
            # Save static image
            plt.savefig(
                "{}Variable_{}_SynthVAE_{}.png".format(
                    saving_filepath, column, pre_proc_method
                )
            )

        plt.show()

    for column in continuous_columns:

        plt.subplot(1, 2, 1)
        plt.hist(x=synthetic_supp[column])
        plt.title("Synthetic")
        # Set the x axis label of the current axis
        plt.xlabel("Data Value")
        # Set the y axis label of the current axis.
        plt.ylabel("Distribution")
        # Set a title of the current axes.
        plt.title("Synthetic".format(column))
        # show a legend on the plot
        plt.subplot(1, 2, 2)
        plt.hist(x=data_supp[column])
        plt.title("Original")
        # Set the x axis label of the current axis
        plt.xlabel("Data Value")
        # Set the y axis label of the current axis.
        plt.ylabel("Distribution")
        # Set a title of the current axes.
        plt.title("Original".format(column))
        # show a legend on the plot
        plt.suptitle("Variable {}".format(column))

        plt.tight_layout()

        if saving_filepath != None:
            # Save static image
            plt.savefig(
                "{}Variable_{}_SynthVAE_{}.png".format(
                    saving_filepath, column, pre_proc_method
                )
            )

        plt.show()

        return None
