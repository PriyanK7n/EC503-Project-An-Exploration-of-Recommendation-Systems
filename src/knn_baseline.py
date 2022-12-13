import pandas as pd
from surprise import Reader, SVD, SVDpp, Dataset, accuracy, similarities, KNNBasic
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
from surprise.model_selection.split import PredefinedKFold
import pickle as pkl
import random
import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from numpy import ndarray

# Load the movielens-100k dataset (download it if needed),

# sample random trainset and testset
# test set is made of 25% of the ratings.
# trainset, testset = train_test_split(data, test_size=0.25)

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

    return final_rating

def increase_sparsity():
    book_df = load_data()
    book_df = book_df[book_df['rating'] != 0]
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(book_df[['user_id', 'title', 'rating']], reader)
    raw_ratings = data.raw_ratings
    random.shuffle(raw_ratings)
    threshold = int(0.8 * len(raw_ratings))
    train_raw_ratings = raw_ratings[:threshold]
    test_raw_ratings = raw_ratings[threshold:]

    trainset = data.construct_trainset(train_raw_ratings)
    testset = data.construct_testset(test_raw_ratings)

    metric_dict = {'mse': [], 'rmse': [], 'mae': [], 'cs': [], 'pr': []}
    test_metric_dict = {'mse': [], 'rmse': [], 'mae': [], 'cs': [], 'pr': []}

    sparsity_list = []

    # test_original = test.copy()
    # train_original = train.copy()
    for i in range(20):
        if i > 0:
            threshold = int(0.8 * (len(raw_ratings)-400*i))
            train_raw_ratings = raw_ratings[i*400:i*400+threshold]
            test_raw_ratings = raw_ratings[i*400+threshold:]

            trainset = data.construct_trainset(train_raw_ratings)
            testset = data.construct_testset(test_raw_ratings)
        #     non_zero_index = np.where(train > 0)
        #     choices = np.random.choice(len(non_zero_index[0])-1, size=300, replace=False)
        #     for j,k in zip(non_zero_index[0][choices], non_zero_index[1][choices]):
        #         train.iloc[j, k] = 0
        #     test_non_zero_index = np.where(test > 0)
        #     test_choices = np.random.choice(len(test_non_zero_index[0])-1, size=75, replace=False)
        #     for j,k in zip(test_non_zero_index[0][test_choices], test_non_zero_index[1][test_choices]):
        #         test.iloc[j, k] = 0
        # book_sparse = csr_matrix(train)
        # print("Sparsity ratio ", train.stack().value_counts()[0] / sum(train.stack().value_counts()))
        # sparsity_list.append(train.stack().value_counts()[0] / sum(train.stack().value_counts()))
        sparsity_list.append(1 - trainset.n_ratings/trainset.n_items/trainset.n_items)

        bsl_options = {
            "method": "sgd",
            "n_epochs": 20,
        }

        sim_options = {"name": "cosine", "user_based": False, "min_support": 5}

        knn_cv_model = KNNBasic(bsl_options=bsl_options, sim_options=sim_options, k=10)
        knn_cv_model.fit(trainset)
        predictions_cv_pp = knn_cv_model.test(testset)

        ratings_pp = np.array([[true_r, est] for (_, _, true_r, est, _) in predictions_cv_pp])
        test_metric_dict['mse'].append(accuracy.mse(predictions_cv_pp))
        test_metric_dict['rmse'].append(accuracy.rmse(predictions_cv_pp))
        test_metric_dict['mae'].append(accuracy.mae(predictions_cv_pp))
        test_metric_dict['cs'].append(cosine(ratings_pp[:, 0], ratings_pp[:, 1]))
        test_metric_dict['pr'].append(pearsonr(ratings_pp[:, 0], ratings_pp[:, 1]).statistic)
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
    return test_metric_dict, sparsity_list

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
    book_pivot = book_pivot.pivot_table(columns='user_id', index='title', values= 'rating')
    # book_pivot.fillna(0, inplace=True)
    non_zero_index = np.where(pd.notna(book_pivot))
    for i in range(book_pivot.shape[1]):
        book_pivot.iloc[:, i].fillna(book_pivot.iloc[:, i].mean(), inplace=True)
    book_sparse = csr_matrix(book_pivot)
    svd = TruncatedSVD(n_components=250, n_iter=1, random_state=42)
    svd.fit(book_sparse)
    predictions: ndarray = np.dot(svd.transform(book_sparse), svd.components_)
    for j, k in zip(non_zero_index[0], non_zero_index[1]):
        predictions[j, k] = book_pivot.iloc[j, k]
    preictions = noiser(np.array(predictions), 0, 1)

    return pd.DataFrame(predictions, columns=book_pivot.columns, index=book_pivot.index)

def increase_sparsity_artificial():
    book_df = create_dense()
    rating_list = [(r, c, ra) for ((r, c), ra) in book_df.stack(0).iteritems()]
    book_df = pd.DataFrame(rating_list, columns=['title', 'user_id', 'rating'])
    book_df = book_df[book_df['rating'] != 0]
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(book_df[['user_id', 'title', 'rating']], reader)
    raw_ratings = data.raw_ratings
    random.shuffle(raw_ratings)
    threshold = int(0.8 * len(raw_ratings))
    train_raw_ratings = raw_ratings[:threshold]
    test_raw_ratings = raw_ratings[threshold:]

    trainset = data.construct_trainset(train_raw_ratings)
    testset = data.construct_testset(test_raw_ratings)

    metric_dict = {'mse': [], 'rmse': [], 'mae': [], 'cs': [], 'pr': []}
    test_metric_dict = {'mse': [], 'rmse': [], 'mae': [], 'cs': [], 'pr': []}

    print('Dense matrix')
    print('Matrix size', trainset.n_items, trainset.n_users)

    for drop_ratio in [0, 20, 40, 60, 80, 90]:
        print("Drop Ratio", drop_ratio)
        total_after_drop = len(raw_ratings) * (1 - drop_ratio/100) - 1
        threshold = int(0.8 * total_after_drop)
        train_raw_ratings = raw_ratings[int(len(raw_ratings) * drop_ratio/100):int(len(raw_ratings) * drop_ratio/100 + threshold)]
        test_raw_ratings = raw_ratings[int(len(raw_ratings) * drop_ratio/100 + threshold):]

        trainset = data.construct_trainset(train_raw_ratings)
        testset = data.construct_testset(test_raw_ratings)

        bsl_options = {
            "method": "sgd",
            "n_epochs": 20,
        }
        sim_options = {"name": "cosine", "user_based": False, "min_support": 5}
        knn_cv_model = KNNBasic(bsl_options=bsl_options, sim_options=sim_options, k=10)
        knn_cv_model.fit(trainset)
        predictions_cv_pp = knn_cv_model.test(testset)

        ratings_pp = np.array([[true_r, est] for (_, _, true_r, est, _) in predictions_cv_pp])
        test_metric_dict['mse'].append(accuracy.mse(predictions_cv_pp))
        test_metric_dict['rmse'].append(accuracy.rmse(predictions_cv_pp))
        test_metric_dict['mae'].append(accuracy.mae(predictions_cv_pp))
        test_metric_dict['cs'].append(cosine(ratings_pp[:, 0], ratings_pp[:, 1]))
        test_metric_dict['pr'].append(pearsonr(ratings_pp[:, 0], ratings_pp[:, 1]).statistic)
    return test_metric_dict

def decrease_sparsity():
    book_df = load_data()
    book_df = book_df[book_df['rating'] != 0]
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(book_df[['user_id', 'title', 'rating']], reader)
    raw_ratings = data.raw_ratings
    random.shuffle(raw_ratings)
    threshold = int(0.8 * len(raw_ratings))
    train_raw_ratings = raw_ratings[:threshold]
    test_raw_ratings = raw_ratings[threshold:]

    trainset = data.construct_trainset(train_raw_ratings)
    testset = data.construct_testset(test_raw_ratings)

    metric_dict = {'mse': [], 'rmse': [], 'mae': [], 'cs': [], 'pr': []}
    test_metric_dict = {'mse': [], 'rmse': [], 'mae': [], 'cs': [], 'pr': []}

    sparsity_list = []

    # test_original = test.copy()
    # train_original = train.copy()
    for i in range(20):
        if i > 0:
            threshold = int(0.8 * (len(raw_ratings)-400*i))
            train_raw_ratings = raw_ratings[i*400:i*400+threshold]
            test_raw_ratings = raw_ratings[i*400+threshold:]

            trainset = data.construct_trainset(train_raw_ratings)
            testset = data.construct_testset(test_raw_ratings)
        #     non_zero_index = np.where(train > 0)
        #     choices = np.random.choice(len(non_zero_index[0])-1, size=300, replace=False)
        #     for j,k in zip(non_zero_index[0][choices], non_zero_index[1][choices]):
        #         train.iloc[j, k] = 0
        #     test_non_zero_index = np.where(test > 0)
        #     test_choices = np.random.choice(len(test_non_zero_index[0])-1, size=75, replace=False)
        #     for j,k in zip(test_non_zero_index[0][test_choices], test_non_zero_index[1][test_choices]):
        #         test.iloc[j, k] = 0
        # book_sparse = csr_matrix(train)
        # print("Sparsity ratio ", train.stack().value_counts()[0] / sum(train.stack().value_counts()))
        # sparsity_list.append(train.stack().value_counts()[0] / sum(train.stack().value_counts()))
        sparsity_list.append(1 - trainset.n_ratings/trainset.n_items/trainset.n_items)

        bsl_options = {
            "method": "sgd",
            "n_epochs": 20,
        }

        sim_options = {"name": "cosine", "user_based": False, "min_support": 5}

        knn_cv_model = KNNBasic(bsl_options=bsl_options, sim_options=sim_options, k=10)
        knn_cv_model.fit(trainset)
        predictions_cv_pp = knn_cv_model.test(testset)

        ratings_pp = np.array([[true_r, est] for (_, _, true_r, est, _) in predictions_cv_pp])
        test_metric_dict['mse'].append(accuracy.mse(predictions_cv_pp))
        test_metric_dict['rmse'].append(accuracy.rmse(predictions_cv_pp))
        test_metric_dict['mae'].append(accuracy.mae(predictions_cv_pp))
        test_metric_dict['cs'].append(cosine(ratings_pp[:, 0], ratings_pp[:, 1]))
        test_metric_dict['pr'].append(pearsonr(ratings_pp[:, 0], ratings_pp[:, 1]).statistic)
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
    return test_metric_dict, sparsity_list


if __name__ == '__main__':
    # test_metric_dict, sparsity_list = increase_sparsity()
    # with open('increase_original_sparsity_knn.pkl', 'wb') as f:
    #     inc_result = {'test_metric': test_metric_dict, 'sparsity': sparsity_list}
    #     pkl.dump(inc_result, f)
    # fig, axes = plt.subplots(3, 2)
    # fig.suptitle('KNN Performance : Metrics VS Sparsity \n(Book Recommendation Test Dataset)')
    name_dict = {'mse': 'MSE', 'rmse': 'RMSE', 'mae': 'MAE', 'cs': 'Cosine Similarity', 'pr': 'Pearson R'}
    # # # # #
    # # middle = inc_sparsity_list[0]
    # # inc_sparsity_list.extend(dec_sparsity_list[1:])
    # # inc_sparsity_list = sorted(inc_sparsity_list)
    # (inc_test_metric_dict, inc_sparsity_list) = (inc_result['test_metric'], inc_result['sparsity'])
    # for i, metric in zip(range(5), ['mse', 'rmse', 'mae', 'cs', 'pr']):
    #     # metric_list = inc_metric_dict[metric]
    #     # metric_list.extend(dec_metric_dict[metric][1:])
    #     # metric_list = [x for _, x in sorted(zip(inc_sparsity_list, metric_list))]
    #     axes.flatten()[i].plot(inc_sparsity_list, inc_test_metric_dict[metric], label='Increasing', linestyle='dashed')
    #     # axes.flatten()[i].plot(dec_sparsity_list, dec_test_metric_dict[metric], label='Decreasing', linestyle='dashed')
    #     # axes.flatten()[i].plot(sorted(inc_sparsity_list), metric_list, label='train')
    #     # axes.flatten()[i].axvline(middle, color='red')
    #     # axes.flatten()[i].plot(sparsity_list, test_metric_dict[metric], linestyle='dashed', label='test')
    #     # axes.flatten()[i].set_xticks(['0%', '20%', '40%', '60%', '80%'])
    #     axes.flatten()[i].set(ylabel=name_dict[metric])
    #     axes.flatten()[i].set(xlabel='Sparsity Ratio')
    #     axes.flatten()[i].legend()
    # fig.delaxes(axes.flatten()[-1])
    # plt.show()

    sparse_test_metric = increase_sparsity_artificial()
    with open('../artifacts/increase_artificial_sparsity_knn.pkl', 'wb') as f:
    # inc_result = {'test_metric': sparse_test_metric}
        pkl.dump(sparse_test_metric, f)
    fig, axes = plt.subplots(3, 2)
    fig.suptitle("KNN Performance : Metrics VS Sparsity \n(Artificial Test Dataset)")
    for i, metric in zip(range(5), ['mse', 'rmse', 'mae', 'cs', 'pr']):
        # axes.flatten()[i].plot([0, 0.2, 0.4, 0.6, 0.8], sparse_metric[metric], label='train')
        axes.flatten()[i].plot([0, 0.2, 0.4, 0.6, 0.8, 0.9], sparse_test_metric[metric], linestyle='dashed', label='Test')
        axes.flatten()[i].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 0.9])
        axes.flatten()[i].set(ylabel=name_dict[metric])
        axes.flatten()[i].set(xlabel="Sparsity")
        axes.flatten()[i].legend()
    fig.delaxes(axes.flatten()[-1])
    plt.show()

