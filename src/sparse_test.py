import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import ndarray
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import pickle as pkl
from numpy.random import default_rng
from sklearn.model_selection import train_test_split

# Data Pre-Processing
def load_data():
    books = pd.read_csv('data/BX-Books.csv', sep=";", on_bad_lines='skip', encoding='latin-1', low_memory=False)
    books = books[['ISBN','Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher','Image-URL-L']]
    # Rename weird columns name
    books.rename(columns={"Book-Title":'title',
                          'Book-Author':'author',
                         "Year-Of-Publication":'year',
                         "Publisher":"publisher",
                         "Image-URL-L":"image_url"},inplace=True)
    # Load the second dataframe
    users = pd.read_csv('data/BX-Users.csv', sep=";", on_bad_lines='skip', encoding='latin-1')
    # Remane some wierd columns name
    users.rename(columns={"User-ID":'user_id',
                          'Location':'location',
                         "Age":'age'},inplace=True)
    # Now load the third dataframe
    ratings = pd.read_csv('data/BX-Book-Ratings.csv', sep=";", on_bad_lines='skip', encoding='latin-1')
    # Lets remane some wierd columns name
    ratings.rename(columns={"User-ID":'user_id',
                          'Book-Rating':'rating'},inplace=True)
    # print(books.shape, users.shape, ratings.shape, sep='\n')
    ratings['user_id'].value_counts()
    # Lets store users who had at least rated more than 300 books
    x = ratings['user_id'].value_counts() > 300
    y= x[x].index
    ratings = ratings[ratings['user_id'].isin(y)]
    ratings_with_books = ratings.merge(books, on='ISBN')
    number_rating = ratings_with_books.groupby('title')['rating'].count().reset_index()
    number_rating.rename(columns={'rating':'num_of_rating'},inplace=True)
    final_rating = ratings_with_books.merge(number_rating, on='title')
    # Lets take those books which got at least 50 rating of user
    final_rating = final_rating[final_rating['num_of_rating'] >= 50]
    # lets drop the duplicates
    final_rating.drop_duplicates(['user_id','title'],inplace=True)
    # Lets create a pivot table
    book_pivot = final_rating.pivot_table(columns='user_id', index='title', values= 'rating')
    # book_pivot.fillna(0, inplace=True)
    return book_pivot

# Increasing Sparsity Test
def increase_sparsity():
    book_pivot = load_data()
    book_pivot.fillna(0, inplace=True)
    metric_dict = {'mse': [], 'rmse': [], 'mae': [], 'cs': [], 'pr': []}
    test_metric_dict = {'mse': [], 'rmse': [], 'mae': [], 'cs': [], 'pr': []}

    sparsity_list = []
    train = book_pivot.sample(frac=0.8, random_state=200)
    test = book_pivot.drop(train.index)
    test_original = test.copy()
    train_original = train.copy()
    for i in range(20):
        if i > 0:
            non_zero_index = np.where(train > 0)
            choices = np.random.choice(len(non_zero_index[0])-1, size=300, replace=False)
            for j,k in zip(non_zero_index[0][choices], non_zero_index[1][choices]):
                train.iloc[j, k] = 0
            test_non_zero_index = np.where(test > 0)
            test_choices = np.random.choice(len(test_non_zero_index[0])-1, size=75, replace=False)
            for j,k in zip(test_non_zero_index[0][test_choices], test_non_zero_index[1][test_choices]):
                test.iloc[j, k] = 0
        book_sparse = csr_matrix(train)
        print("Sparsity ratio ", train.stack().value_counts()[0] / sum(train.stack().value_counts()))
        sparsity_list.append(train.stack().value_counts()[0] / sum(train.stack().value_counts()))
        svd = TruncatedSVD(n_components=250, n_iter=20, random_state=42)
        svd.fit(book_sparse)
        predictions = np.dot(svd.transform(book_sparse), svd.components_)

        true = np.squeeze(np.array(book_sparse.todense().flatten()))
        non_zero_index = np.where(true > 0)
        true = true[non_zero_index]
        # predictions = predictions.flatten()[non_zero_index]
        train_index = np.where(np.array(train_original).flatten() > 0)
        mse, rmse, mae, cs, pr = get_metrics(np.array(train_original).flatten()[train_index], predictions.flatten()[train_index])
        test_index = np.array(test_original).flatten() > 0
        predictions_t = np.dot(svd.transform(test), svd.components_)
        t_mse, t_rmse, t_mae, t_cs, t_pr = get_metrics(np.array(test_original).flatten()[test_index], predictions_t.flatten()[test_index])
        metric_dict['mse'].append(mse)
        metric_dict['rmse'].append(rmse)
        metric_dict['mae'].append(mae)
        metric_dict['cs'].append(cs)
        metric_dict['pr'].append(pr)
        test_metric_dict['mse'].append(t_mse)
        test_metric_dict['rmse'].append(t_rmse)
        test_metric_dict['mae'].append(t_mae)
        test_metric_dict['cs'].append(t_cs)
        test_metric_dict['pr'].append(t_pr)
    # metric_dict = {'MSE': mse_list, 'RMSE': rmse_list, 'MAE': mae_list, 'Cosine Similarity': cs_list, 'Pearson R': pr_list}
    # test_metric_dict = {'MSE': t_mse_list, 'RMSE': t_rmse_list, 'MAE': t_mae_list, 'Cosine Similarity': t_cs_list, 'Pearson R': t_pr_list}
    # plt.figure()
    # fig, axes = plt.subplots(3, 2, sharex=True)
    # axes = axes.flatten()
    # for i, metric in zip(range(5), ['MSE', 'RMSE', 'MAE', 'Cosine Similarity', 'Pearson R']):
    #     axes[i].plot(range(20), metric_dict[metric])
    #     axes[i].set(ylabel=metric)
    #     axes[i].set(xlabel='Iterations')
    #     axes[i].grid()
    # fig.suptitle('Plots of evaluation metrics trained on dataset with increasing sparsity', fontsize=16)
    # plt.show()
    return metric_dict, test_metric_dict, sparsity_list

# Decreasing Sparsity Test
def decrease_sparsity():
    book_pivot = load_data()
    book_pivot.fillna(0, inplace=True)
    metric_dict = {'mse': [], 'rmse': [], 'mae': [], 'cs': [], 'pr': []}
    test_metric_dict = {'mse': [], 'rmse': [], 'mae': [], 'cs': [], 'pr': []}

    sparsity_list = []
    train = book_pivot.sample(frac=0.8, random_state=200)
    test = book_pivot.drop(train.index)
    for i in range(20):
        if i > 0:
            zero_index = np.where(train == 0)
            choices = np.random.choice(len(zero_index[0]) - 1, size=300, replace=False)
            for j, k in zip(zero_index[0][choices], zero_index[1][choices]):
                train.iloc[j, k] = np.mean(train.iloc[:,k])
            test_zero_index = np.where(test == 0)
            choices = np.random.choice(len(test_zero_index[0]) - 1, size=75, replace=False)
            for j, k in zip(test_zero_index[0][choices], test_zero_index[1][choices]):
                test.iloc[j, k] = np.mean(test.iloc[:,k])
        book_sparse = csr_matrix(train)
        print("Sparsity ratio ", train.stack().value_counts()[0] / sum(train.stack().value_counts()))
        sparsity_list.append(train.stack().value_counts()[0] / sum(train.stack().value_counts()))
        svd = TruncatedSVD(n_components=250, n_iter=20, random_state=42)
        svd.fit(book_sparse)
        predictions = np.dot(svd.transform(book_sparse), svd.components_)

        true = np.squeeze(np.array(book_sparse.todense().flatten()))
        non_zero_index = np.where(true > 0)
        true = true[non_zero_index]
        predictions = predictions.flatten()[non_zero_index]
        mse, rmse, mae, cs, pr = get_metrics(true, predictions)
        test_index = np.array(test).flatten() > 0
        predictions_t = np.dot(svd.transform(test), svd.components_)
        t_mse, t_rmse, t_mae, t_cs, t_pr = get_metrics(np.array(test).flatten()[test_index], predictions_t.flatten()[test_index])
        metric_dict['mse'].append(mse)
        metric_dict['rmse'].append(rmse)
        metric_dict['mae'].append(mae)
        metric_dict['cs'].append(cs)
        metric_dict['pr'].append(pr)
        test_metric_dict['mse'].append(t_mse)
        test_metric_dict['rmse'].append(t_rmse)
        test_metric_dict['mae'].append(t_mae)
        test_metric_dict['cs'].append(t_cs)
        test_metric_dict['pr'].append(t_pr)
    return metric_dict, test_metric_dict, sparsity_list


# Cross Validation/Hyper Parameter Tuning
def cross_validation():

    def custom_scorer(estimator, X, y):
        predictions = np.dot(estimator.transform(X), estimator.components_)
        true = np.squeeze(np.array(y.todense().flatten()))
        non_zero_index = np.where(true > 0)
        true = true[non_zero_index]
        predictions = predictions.flatten()[non_zero_index]
        mse = mean_squared_error(true, predictions)
        return -math.sqrt(mse)

    book_pivot = load_data()
    book_sparse = csr_matrix(book_pivot)
    # parameters = {'n_components': [10 * i for i in range(1, 50)], 'n_iter': [10 * i for i in range(1, 50)], 'algorithm': ['arpack', 'randomized']}
    parameters = {'n_components': [10 * i for i in range(1, 56)], 'n_iter': [10 * i for i in range(1, 4)], 'algorithm': [ 'randomized']}
    clf = GridSearchCV(TruncatedSVD(), parameters, scoring=custom_scorer, verbose=3, n_jobs=-1)
    clf.fit(book_sparse, book_sparse)

    sorted(clf.cv_results_.keys())
    print("best parameters", clf.best_params_)

def get_metrics(true, predictions):
    mse_list = []
    rmse_list = []
    mae_list = []
    cs_list = []
    pr_list = []

    # non_zero_index = np.where(true > 0)
    # true = true[non_zero_index]
    # predictions = predictions.flatten()[non_zero_index]
    mse = mean_squared_error(true, predictions)
    mse_list.append(mse)
    print("MSE: ", mse)
    rmse = math.sqrt(mse)
    rmse_list.append(rmse)
    print("RMSE: ", rmse)
    mae = mean_absolute_error(true, predictions)
    mae_list.append(mae)
    print("MAE: ", mae)
    # cs = np.dot(true, predictions) / np.linalg.norm(true) / np.linalg.norm(predictions)
    cs = cosine(true, predictions)
    cs_list.append(cs)
    print("Cosine Similarity: ", cs)
    pr = pearsonr(true, predictions).statistic
    pr_list.append(pr)
    print("PearsonR: ", pr)
    print()
    return mse, rmse, mae, cs, pr

def compare_mean_median():
    book_pivot = load_data()
    zero_index = np.where(book_pivot == 0)
    for j, k in zip(zero_index[0], zero_index[1]):
        book_pivot.iloc[j, k] = np.median(book_pivot.iloc[:, k])
    book_sparse = csr_matrix(book_pivot)
    svd = TruncatedSVD(n_components=250, n_iter=20, random_state=42)
    svd.fit(book_sparse)
    predictions = np.dot(svd.transform(book_sparse), svd.components_)
    true = np.squeeze(np.array(book_sparse.todense().flatten()))
    mse_list, rmse_list, mae_list, cs_list, pr_list = get_metrics(true, predictions)

def plot_svd_iteration():
    metric_dict = {}
    test_metric_dict = {}
    fig, axes = plt.subplots(3, 2)
    original = load_data()
    train = original.sample(frac=0.8, random_state=200)
    test = original.drop(train.index)
    for method in ['mean_row', 'mean_col', 'median_row', 'median_col', 'zero']:
        print(method)
        book_pivot = train.copy()
        test_pivot = test.copy()
        non_zero_index = np.where(pd.notna(train))
        train_evaluate_index = np.where(np.array(pd.notna(train)).flatten())
        test_evaluate_index = np.where(np.array(pd.notna(test)).flatten())
        if method == 'mean_row':
            for i in range(book_pivot.shape[0]):
                book_pivot.iloc[i, :].fillna(book_pivot.iloc[i, :].mean(), inplace=True)
            for i in range(test_pivot.shape[0]):
                test_pivot.iloc[i, :].fillna(test_pivot.iloc[i, :].mean(), inplace=True)
        elif method == 'mean_col':
            for i in range(book_pivot.shape[1]):
                book_pivot.iloc[:, i].fillna(book_pivot.iloc[:, i].mean(), inplace=True)
            for i in range(test_pivot.shape[1]):
                test_pivot.iloc[:, i].fillna(test_pivot.iloc[:, i].mean(), inplace=True)
        elif method == 'median_row':
            for i in range(book_pivot.shape[0]):
                book_pivot.iloc[i, :].fillna(book_pivot.iloc[i, :].median(), inplace=True)
            for i in range(test_pivot.shape[0]):
                test_pivot.iloc[i, :].fillna(test_pivot.iloc[i, :].median(), inplace=True)
        elif method == 'median_col':
            for i in range(book_pivot.shape[1]):
                book_pivot.iloc[:, i].fillna(book_pivot.iloc[:, i].median(), inplace=True)
            for i in range(test_pivot.shape[1]):
                test_pivot.iloc[:, i].fillna(test_pivot.iloc[:, i].median(), inplace=True)
        # elif method == 'zero':
        book_pivot.fillna(0, inplace=True)
        test_pivot.fillna(0, inplace=True)
        for i in range(5):
            print(i)
            book_sparse = csr_matrix(book_pivot)
            svd = TruncatedSVD(n_components=250, n_iter=1, random_state=42)
            svd.fit(book_sparse)
            predictions = np.dot(svd.transform(book_sparse), svd.components_)
            test_predictions = np.dot(svd.transform(test_pivot), svd.components_)
            true = np.squeeze(np.array(book_sparse.todense().flatten()))
            print('Train')
            mse, rmse, mae, cs, pr = get_metrics(true[train_evaluate_index], predictions.flatten()[train_evaluate_index])
            print('Test')
            t_mse, t_rmse, t_mae, t_cs, t_pr = get_metrics(np.array(test).flatten()[test_evaluate_index], test_predictions.flatten()[test_evaluate_index])
            if method not in metric_dict.keys():
                metric_dict[method] = {'mse': [mse], 'rmse': [rmse], 'mae': [mae], 'cs': [cs], 'pr': [pr]}
                test_metric_dict[method] = {'mse': [t_mse], 'rmse': [t_rmse], 'mae': [t_mae], 'cs': [t_cs], 'pr': [t_pr]}
            else:
                metric_dict[method]['mse'].append(mse)
                metric_dict[method]['rmse'].append(rmse)
                metric_dict[method]['mae'].append(mae)
                metric_dict[method]['cs'].append(cs)
                metric_dict[method]['pr'].append(pr)
                test_metric_dict[method]['mse'].append(t_mse)
                test_metric_dict[method]['rmse'].append(t_rmse)
                test_metric_dict[method]['mae'].append(t_mae)
                test_metric_dict[method]['cs'].append(t_cs)
                test_metric_dict[method]['pr'].append(t_pr)
            book_pivot = pd.DataFrame(predictions)
            for j, k in zip(non_zero_index[0], non_zero_index[1]):
                book_pivot.iloc[j, k] = train.iloc[j, k]
    return metric_dict, test_metric_dict


def create_dense():
    def noiser(mat, mean, var, round_toggle=True, bound_toggle=True):
        # mat => input matrix
        # mean => mean of noise
        # var => variance of noise
        # round_toggle => toggles rounding of result to whole numbers on/off
        # bound_toggle => toggles bounding of result to input matrix min & max on/off

        # Generate Noise
        noise = np.random.normal(mean, var, np.shape(mat))
        # Add noise to input matrix
        mat_new = mat + noise

        if (round_toggle == True):
            # round to nearest half (0.5)
            mat_new = np.round(mat_new)

        if (bound_toggle == True):
            mat_new[mat_new < np.min(mat)] = np.min(mat)
            mat_new[mat_new > np.max(mat)] = np.max(mat)

        return mat_new

    book_pivot = load_data()
    non_zero_index = np.where(pd.notna(book_pivot))
    for i in range(book_pivot.shape[1]):
        book_pivot.iloc[:, i].fillna(book_pivot.iloc[:, i].mean(), inplace=True)
    book_sparse = csr_matrix(book_pivot)
    svd = TruncatedSVD(n_components=250, n_iter=1, random_state=42)
    svd.fit(book_sparse)
    predictions: ndarray = np.dot(svd.transform(book_sparse), svd.components_)
    for j, k in zip(non_zero_index[0], non_zero_index[1]):
        predictions[j, k] = book_pivot.iloc[j, k]
    preictions = noiser(np.array(predictions), 0, 2)
    return preictions

def show_plot(metric_dict, test_metric_dict):
    name_dict = {'mse': 'MSE', 'rmse': 'RMSE', 'mae': 'MAE', 'cs': 'Cosine Similarity', 'pr': 'Pearson R'}
    fig, axes = plt.subplots(3, 2)
    fig.suptitle('Iteration Results of Training Data\n(Book Recommendation Dataset)')
    for i, metric in zip(range(5), ['mse', 'rmse', 'mae', 'cs', 'pr']):
        for method in ['mean_row', 'mean_col', 'median_row', 'median_col', 'zero']:
            axes.flatten()[i].plot(metric_dict[method][metric], label=method)
            # axes.flatten()[i].plot(test_metric_dict[method][metric], linestyle='dashed', label='test_'+method)
            axes.flatten()[i].set(ylabel=name_dict[metric])
            axes.flatten()[i].set(xlabel='Iterations')
        axes.flatten()[i].legend()
    fig.delaxes(axes.flatten()[-1])
    fig, axes = plt.subplots(3, 2)
    fig.suptitle('Iteration Results of Test Data\n(Book Recommendation Dataset)')
    for i, metric in zip(range(5), ['mse', 'rmse', 'mae', 'cs', 'pr']):
        for method in ['mean_row', 'mean_col', 'median_row', 'median_col', 'zero']:
            # axes.flatten()[i].plot(test_metric_dict[method][metric], label='test_'+method)
            axes.flatten()[i].plot(test_metric_dict[method][metric], linestyle='dashed', label=method)
            axes.flatten()[i].set(ylabel=name_dict[metric])
            axes.flatten()[i].set(xlabel='Iterations')
        axes.flatten()[i].legend()
    fig.delaxes(axes.flatten()[-1])
    plt.show()

def test_artificial():
    dense_matrix = create_dense()
    book_pivot = pd.DataFrame(dense_matrix)
    print('Dense matrix')
    book_sparse = csr_matrix(book_pivot)
    svd = TruncatedSVD(n_components=250, n_iter=1, random_state=42)
    svd.fit(book_sparse)
    predictions = np.dot(svd.transform(book_sparse), svd.components_)
    true = np.squeeze(np.array(book_sparse.todense().flatten()))
    mse, rmse, mae, cs, pr = get_metrics(true, predictions.flatten())

    print('dropped 10% rows and 50% columns')
    drop_data = book_pivot.copy()
    arr_indices_top_drop = default_rng().choice(drop_data.index, size=50, replace=False)
    drop_data.drop(index=arr_indices_top_drop, inplace=True)
    arr_indices_top_drop = default_rng().choice(drop_data.index, size=275, replace=False)
    drop_data.drop(columns=arr_indices_top_drop, inplace=True)
    book_sparse = csr_matrix(drop_data)
    svd = TruncatedSVD(n_components=250, n_iter=1, random_state=42)
    svd.fit(book_sparse)
    predictions = np.dot(svd.transform(book_sparse), svd.components_)
    true = np.squeeze(np.array(book_sparse.todense().flatten()))
    mse, rmse, mae, cs, pr = get_metrics(true, predictions.flatten())

    print('dropped 20% rows and 20% columns')
    drop_data = book_pivot.copy()
    arr_indices_top_drop = default_rng().choice(drop_data.index, size=100, replace=False)
    drop_data.drop(index=arr_indices_top_drop, inplace=True)
    arr_indices_top_drop = default_rng().choice(drop_data.index, size=110, replace=False)
    drop_data.drop(columns=arr_indices_top_drop, inplace=True)
    book_sparse = csr_matrix(drop_data)
    svd = TruncatedSVD(n_components=250, n_iter=1, random_state=42)
    svd.fit(book_sparse)
    predictions = np.dot(svd.transform(book_sparse), svd.components_)
    true = np.squeeze(np.array(book_sparse.todense().flatten()))
    mse, rmse, mae, cs, pr = get_metrics(true, predictions.flatten())

def increase_sparsity_artificial():
    metric_dict = {'mse': [], 'rmse': [], 'mae': [], 'cs': [], 'pr': []}
    test_metric_dict = {'mse': [], 'rmse': [], 'mae': [], 'cs': [], 'pr': []}
    dense_matrix = create_dense()
    original = pd.DataFrame(dense_matrix)
    # original = pd.DataFrame(np.random.randint(0, 10.5, (500, 551)))
    drop_data = original.copy()
    arr_indices_top_drop = default_rng().choice(drop_data.index, size=100, replace=False)
    drop_data.drop(index=arr_indices_top_drop, inplace=True)
    arr_indices_top_drop = default_rng().choice(drop_data.index, size=110, replace=False)
    drop_data.drop(columns=arr_indices_top_drop, inplace=True)
    original = drop_data
    print('Dense matrix')
    print('Matrix size', original.shape)
    train = original.sample(frac=0.8, random_state=200)
    test = original.drop(train.index)
    for drop_ratio in [0, 20, 40, 60, 80, 90]:
        print("Drop Ratio", drop_ratio)
        train_choices = np.random.choice(train.size-1, size=int(drop_ratio / 100 * train.size), replace=False)
        test_choices = np.random.choice(test.size-1, size=int(drop_ratio / 100 * test.size), replace=False)
        book_pivot = train.copy()
        test_pivot = test.copy()
        for index in train_choices:
            book_pivot.iloc[index // train.shape[1], index % train.shape[0]] = 0
        for index in test_choices:
            test_pivot.iloc[index // test.shape[1], index % test.shape[0]] = 0
        book_sparse = csr_matrix(book_pivot)
        svd = TruncatedSVD(n_components=250, n_iter=20)
        svd.fit(book_sparse)
        predictions = np.dot(svd.transform(book_sparse), svd.components_)
        train_index = np.array(train).flatten() > 0
        mse, rmse, mae, cs, pr = get_metrics(np.array(train).flatten()[train_index], predictions.flatten()[train_index])
        predictions_t = np.dot(svd.transform(test_pivot), svd.components_)
        test_true = np.array(test).flatten()
        test_index = np.array(test).flatten() > 0
        t_mse, t_rmse, t_mae, t_cs, t_pr = get_metrics(test_true[test_index], predictions_t.flatten()[test_index])
        metric_dict['mse'].append(mse)
        metric_dict['rmse'].append(rmse)
        metric_dict['mae'].append(mae)
        metric_dict['cs'].append(cs)
        metric_dict['pr'].append(pr)
        test_metric_dict['mse'].append(t_mse)
        test_metric_dict['rmse'].append(t_rmse)
        test_metric_dict['mae'].append(t_mae)
        test_metric_dict['cs'].append(t_cs)
        test_metric_dict['pr'].append(t_pr)
    return metric_dict, test_metric_dict


def plot_book_dataset():
    with open('decrease_original_sparsity.pkl', 'rb') as f:
        dec_result = pkl.load(f)
    with open('increase_original_sparsity.pkl', 'rb') as f:
        inc_result = pkl.load(f)
    name_dict = {'mse': 'MSE', 'rmse': 'RMSE', 'mae': 'MAE', 'cs': 'Cosine Similarity', 'pr': 'Pearson R'}
    (inc_metric_dict, inc_test_metric_dict, inc_sparsity_list) = (inc_result['train_metric'], inc_result['test_metric'], inc_result['sparsity'])
    (dec_metric_dict, dec_test_metric_dict, dec_sparsity_list) = (dec_result['train_metric'], dec_result['test_metric'], dec_result['sparsity'])
    fig, axes = plt.subplots(3, 2)
    fig.suptitle('SVD Performance : Metrics VS Sparsity \n(Book Recommendation Training Dataset)')
    # # # #
    middle = inc_sparsity_list[0]
    # inc_sparsity_list.extend(dec_sparsity_list[1:])
    # inc_sparsity_list = sorted(inc_sparsity_list)
    for i, metric in zip(range(5), ['mse', 'rmse', 'mae', 'cs', 'pr']):

        # metric_list = inc_metric_dict[metric]
        # metric_list.extend(dec_metric_dict[metric][1:])
        # metric_list = [x for _, x in sorted(zip(inc_sparsity_list, metric_list))]
        axes.flatten()[i].plot(inc_sparsity_list, inc_metric_dict[metric], label='Increasing')
        axes.flatten()[i].plot(dec_sparsity_list, dec_metric_dict[metric], label='Decreasing')
        # axes.flatten()[i].plot(sorted(inc_sparsity_list), metric_list, label='train')
        axes.flatten()[i].axvline(middle, color='red')
        # axes.flatten()[i].plot(sparsity_list, test_metric_dict[metric], linestyle='dashed', label='test')
        # axes.flatten()[i].set_xticks(['0%', '20%', '40%', '60%', '80%'])
        axes.flatten()[i].set(ylabel=name_dict[metric])
        axes.flatten()[i].set(xlabel='Sparsity Ratio')
        axes.flatten()[i].legend()
    fig.delaxes(axes.flatten()[-1])
    plt.show()

    fig, axes = plt.subplots(3, 2)
    fig.suptitle('SVD Performance : Metrics VS Sparsity \n(Book Recommendation Test Dataset)')
    # # # #
    middle = inc_sparsity_list[0]
    # inc_sparsity_list.extend(dec_sparsity_list[1:])
    # inc_sparsity_list = sorted(inc_sparsity_list)
    for i, metric in zip(range(5), ['mse', 'rmse', 'mae', 'cs', 'pr']):

        # metric_list = inc_metric_dict[metric]
        # metric_list.extend(dec_metric_dict[metric][1:])
        # metric_list = [x for _, x in sorted(zip(inc_sparsity_list, metric_list))]
        axes.flatten()[i].plot(inc_sparsity_list, inc_test_metric_dict[metric], label='Increasing', linestyle='dashed')
        axes.flatten()[i].plot(dec_sparsity_list, dec_test_metric_dict[metric], label='Decreasing', linestyle='dashed')
        # axes.flatten()[i].plot(sorted(inc_sparsity_list), metric_list, label='train')
        axes.flatten()[i].axvline(middle, color='red')
        # axes.flatten()[i].plot(sparsity_list, test_metric_dict[metric], linestyle='dashed', label='test')
        # axes.flatten()[i].set_xticks(['0%', '20%', '40%', '60%', '80%'])
        axes.flatten()[i].set(ylabel=name_dict[metric])
        axes.flatten()[i].set(xlabel='Sparsity Ratio')
        axes.flatten()[i].legend()
    fig.delaxes(axes.flatten()[-1])
    plt.show()

if __name__ == '__main__':
    # compare_mean_median()
    # metric_dict, test_metric_dict = plot_svd_iteration()
    # with open('metric_dict5.pkl', 'wb') as f:
    #     pkl.dump(metric_dict, f)
    # with open('test_metric_dict5.pkl', 'wb') as f:
    #     pkl.dump(test_metric_dict, f)
    # show_plot(metric_dict, test_metric_dict)
    # with open('metric_dict5.pkl', 'rb') as f:
    #     metric_dict = pkl.load(f)
    # with open('test_metric_dict5.pkl', 'rb') as f:
    #     test_metric_dict = pkl.load(f)
    #
    # show_plot(metric_dict, test_metric_dict)
    # show_plot(test_metric_dict, test_metric_dict)
    # test_artificial()
    # original dataset
    # metric_dict, test_metric_dict, sparsity_list = increase_sparsity()
    # with open('increase_original_sparsity.pkl', 'wb') as f:
    #     result = {'train_metric': metric_dict, 'test_metric': test_metric_dict, 'sparsity': sparsity_list}
    #     pkl.dump(result, f)
    # metric_dict, test_metric_dict, sparsity_list = decrease_sparsity()
    # with open('decrease_original_sparsity.pkl', 'wb') as f:
    #     result = {'train_metric': metric_dict, 'test_metric': test_metric_dict, 'sparsity': sparsity_list}
    #     pkl.dump(result, f)

    # decrease_sparsity()
    # cross_validation()

    plot_book_dataset()
    # artificial dataset
    # sparse_metric, sparse_test_metric = increase_sparsity_artificial()
    # with open('sparse_artificial_metric_dict20.pkl', 'wb') as f:
    #     pkl.dump(sparse_metric, f)
    # with open('sparse_artificial_test_metric_dict20.pkl', 'wb') as f:
    #     pkl.dump(sparse_test_metric, f)
    # #
    # with open('sparse_artificial_metric_dict20.pkl', 'rb') as f:
    #     sparse_metric = pkl.load(f)
    # with open('sparse_artificial_test_metric_dict20.pkl', 'rb') as f:
    #     sparse_test_metric = pkl.load(f)
    #
    # name_dict = {'mse':'MSE', 'rmse': 'RMSE', 'mae': 'MAE', 'cs': 'Cosine Similarity', 'pr': 'Pearson R'}
    # fig, axes = plt.subplots(3, 2)
    # fig.suptitle("SVD Performance : Metrics VS Sparsity \n(Artificial Train Dataset)")
    # for i, metric in zip(range(5), ['mse', 'rmse', 'mae', 'cs', 'pr']):
    #     axes.flatten()[i].plot([0, 0.2, 0.4, 0.6, 0.8, 0.9], sparse_metric[metric], label='train')
    #     # axes.flatten()[i].plot([0, 0.2, 0.4, 0.6, 0.8], sparse_test_metric[metric], linestyle='dashed', label='test')
    #     axes.flatten()[i].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 0.9])
    #     axes.flatten()[i].set(ylabel=name_dict[metric])
    #     axes.flatten()[i].set(xlabel="Sparsity")
    #     axes.flatten()[i].legend()
    # fig.delaxes(axes.flatten()[-1])
    #
    # fig, axes = plt.subplots(3, 2)
    # fig.suptitle("SVD Performance : Metrics VS Sparsity \n(Artificial Test Dataset)")
    # for i, metric in zip(range(5), ['mse', 'rmse', 'mae', 'cs', 'pr']):
    #     # axes.flatten()[i].plot([0, 0.2, 0.4, 0.6, 0.8], sparse_metric[metric], label='train')
    #     axes.flatten()[i].plot([0, 0.2, 0.4, 0.6, 0.8, 0.9], sparse_test_metric[metric], linestyle='dashed', label='Test')
    #     axes.flatten()[i].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 0.9])
    #     axes.flatten()[i].set(ylabel=name_dict[metric])
    #     axes.flatten()[i].set(xlabel="Sparsity")
    #     axes.flatten()[i].legend()
    # fig.delaxes(axes.flatten()[-1])
    # plt.show()