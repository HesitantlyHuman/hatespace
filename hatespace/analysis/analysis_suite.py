import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import lsq_linear, linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy.special import psi, polygamma, betainc, gamma
import ast
import matplotlib.pyplot as plt
from scipy import special
import time
import logging
import seaborn as sns
import threading
from typing import Optional, Tuple, Union
from hatespace.datasets import IronMarch
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from hatespace.analysis.dirichlet_tools import DirichletGOF
import os
from datetime import datetime, timedelta
import warnings
import itertools
import shutil


def get_range_of_days(
    range_center: Union[int, Tuple[int, int, int]], days: int
) -> Tuple[int, int]:
    if isinstance(range_center, (list, tuple)):
        range_center = datetime(*range_center)
    else:
        range_center = datetime.fromtimestamp(range_center)
    time_range = timedelta(days=days)
    start_time = range_center - time_range
    end_time = range_center + time_range
    range_center = time.mktime(range_center.timetuple())
    start_time = int(time.mktime(start_time.timetuple()))
    end_time = int(time.mktime(end_time.timetuple()))
    return start_time, end_time


class IronmarchAnalysis:
    def __init__(
        self,
        dataset_path: str,
        dataset=None,
        latent_vectors_file_path: Optional[str] = "",
        latent_vectors: Optional[np.ndarray] = None,
        values_dict: Optional[dict] = {},
    ):

        self.dataset_path = dataset_path

        if dataset is None:
            self.dataset = IronMarch(dataset_path)
        else:
            self.dataset = dataset

        self.members = pd.read_csv(os.path.join(dataset_path, "core_members.csv"))

        if values_dict != {}:
            self.values_dict = values_dict

            self.latent_vectors_list = values_dict["data"]["latent_vectors"]
            self.posts_list = values_dict["data"]["posts"]
            self.authors_list = values_dict["data"]["authors"]
            self.ymd_timestamps_list = values_dict["data"]["ymd_timestamps"]
            self.post_ids_list = values_dict["data"]["post_ids"]

            if (
                len(self.latent_vectors_list) == 0
                or self.latent_vectors_list[0].shape[0] == 0
            ):
                raise ValueError(
                    "No posts with specified conditions. Specified time range is outside the original time range and/or given author(s) have no posts in time range."
                )

            self.latent_vectors = np.concatenate(self.latent_vectors_list, axis=0)
            self.posts = list(itertools.chain(*self.posts_list))
            self.post_ids = list(itertools.chain(*self.post_ids_list))

            self.forums = values_dict["forums"]
            self.msgs = values_dict["msgs"]

            self.id_index_dict = dict(
                (value, idx) for idx, value in enumerate(self.post_ids)
            )

        else:

            if latent_vectors_file_path is not "":
                try:
                    self.latent_vectors = np.load(latent_vectors_file_path)
                except:
                    print("Latent vectors file must by .npy file")
            elif latent_vectors is None:
                raise ValueError(
                    "Must provide path to a latent vectors file, or provide a latent vectors array directly"
                )

            self.posts = [x["data"] for x in self.dataset]
            self.posts = [str(i) for i in self.posts]

            self.post_ids = [x["id"] for x in self.dataset]
            self.id_index_dict = dict(
                (value, idx) for idx, value in enumerate(self.post_ids)
            )

            self.forums, self.msgs = self.return_df_from_ids(
                forums=pd.read_csv(os.path.join(dataset_path, "core_search_index.csv")),
                msgs=pd.read_csv(os.path.join(dataset_path, "core_message_posts.csv")),
                post_ids=self.post_ids,
            )

            (
                latent_vectors,
                unix_timestamps,
                ymd_timestamps,
                post_ids,
                posts,
                authors,
            ) = self.return_sorted(self.forums, self.msgs)

            self.values_dict = {
                "data": self.make_data_dict(
                    [latent_vectors],
                    [unix_timestamps],
                    [ymd_timestamps],
                    [post_ids],
                    [posts],
                    [authors],
                ),
                "forums": self.forums,
                "msgs": self.msgs,
            }

            self.latent_vectors = latent_vectors
            self.posts = posts
            self.post_ids = post_ids
            self.id_index_dict = dict(
                (value, idx) for idx, value in enumerate(self.post_ids)
            )

            self.latent_vectors_list = [latent_vectors]
            self.posts_list = [posts]
            self.authors_list = [authors]
            self.ymd_timestamps_list = [ymd_timestamps]
            self.post_ids_list = [post_ids]

        self.latent_dim_size = self.latent_vectors.shape[1]

    def index_by_indices(self, list_to_index, indices):
        return [list_to_index[i] for i in indices]

    def get_posts_from_post_ids(self, post_ids=[]):
        posts = [self.posts[self.id_index_dict[x]] for x in post_ids]
        return posts

    def return_sorted(self, forums, msgs):
        forum_ids = ["forum_post-" + str(x) for x in forums["index_id"].tolist()]
        msg_ids = ["direct_message-" + str(x) for x in msgs["msg_id"].tolist()]

        post_ids = forum_ids + msg_ids
        unix_timestamps = (
            forums["index_date_created"].tolist() + msgs["msg_date"].tolist()
        )
        ymd_timestamps = [
            datetime.fromtimestamp(x).strftime("%Y/%m/%d") for x in unix_timestamps
        ]
        authors = forums["index_author"].tolist() + msgs["msg_author_id"].tolist()

        indices = [self.id_index_dict[x] for x in post_ids]

        latent_vectors = self.latent_vectors[indices]

        sort_indices = np.argsort(unix_timestamps)
        latent_vectors = latent_vectors[sort_indices]
        unix_timestamps = self.index_by_indices(unix_timestamps, sort_indices)
        ymd_timestamps = self.index_by_indices(ymd_timestamps, sort_indices)
        post_ids = self.index_by_indices(post_ids, sort_indices)
        posts = self.get_posts_from_post_ids(post_ids)
        authors = self.index_by_indices(authors, sort_indices)

        return latent_vectors, unix_timestamps, ymd_timestamps, post_ids, posts, authors

    def return_df_from_ids(self, forums, msgs, post_ids):
        forum_ids = [
            int(x.replace("forum_post-", "")) for x in post_ids if "forum_post" in x
        ]
        msg_ids = [
            int(x.replace("direct_message-", ""))
            for x in post_ids
            if "direct_message" in x
        ]

        forums = forums[forums["index_id"].isin(forum_ids)].sort_values(
            by="index_date_created"
        )
        msgs = msgs[msgs["msg_id"].isin(msg_ids)].sort_values(by="msg_date")

        return forums, msgs

    def get(
        self,
        start_time: Optional[Union[Tuple[int, int, int], int]] = None,
        end_time: Optional[Union[Tuple[int, int, int], int]] = None,
        author_ids: Optional[list] = [],
        author_names: Optional[list] = [],
        split_by: str = "",
    ) -> "IronmarchAnalysis":

        if type(start_time) == tuple:
            start = int(
                datetime(start_time[0], start_time[1], start_time[2]).timestamp()
            )
        elif type(start_time) == int:
            start = start_time
        elif start_time == None:
            start = -1
        else:
            warnings.warn(
                "Provide either a tuple (year, month, day) or a UNIX timestamp. start_time is defaulted to beginning of dataset."
            )
            start = -1

        if type(end_time) == tuple:
            end = int(datetime(end_time[0], end_time[1], end_time[2]).timestamp())
        elif type(end_time) == int:
            end = end_time
        elif end_time == None:
            end = 1e99
        else:
            warnings.warn(
                "Provide either a tuple (year, month, day) or a UNIX timestamp. end_time is defaulted to end of dataset."
            )

        forums = self.forums[
            (self.forums["index_date_created"] > start)
            & (self.forums["index_date_created"] < end)
        ]
        msgs = self.msgs[
            (self.msgs["msg_date"] > start) & (self.msgs["msg_date"] < end)
        ]

        if len(author_ids) > 0:
            forums = forums[forums["index_author"].isin(author_ids)]
            msgs = msgs[msgs["msg_author_id"].isin(author_ids)]

        if len(author_names) > 0:
            ids = self.members[self.members['name'].isin(author_names)]['member_id'].tolist()
            forums = forums[forums["index_author"].isin(ids)]
            msgs = msgs[msgs["msg_author_id"].isin(ids)]

        (
            latent_vectors,
            unix_timestamps,
            ymd_timestamps,
            post_ids,
            posts,
            authors,
        ) = self.return_sorted(forums, msgs)

        if split_by != "":

            if split_by.lower() == 'day' or split_by.lower() == 'month':
                if split_by.lower() == "day":
                    dates = [
                        datetime.fromtimestamp(x).strftime("%Y/%m/%d")
                        for x in unix_timestamps
                    ]
                elif split_by.lower() == "month":
                    dates = [
                        datetime.fromtimestamp(x).strftime("%Y/%m") for x in unix_timestamps
                    ]
                current = None
                split_indices = []

                for idx, date in enumerate(dates):
                    if current != date:
                        current = date
                        time_indices = []
                        split_indices.append(time_indices)
                    time_indices.append(idx)

            elif bool(re.match("[0-9]+d$", split_by.lower())):
                delta = timedelta(days=int(split_by[:-1]))

                split_indices = []
                timestamps = [datetime.fromtimestamp(x).replace(hour=0, minute=0, second=0) for x in unix_timestamps]

                initial = datetime.fromtimestamp(start)

                offset = 1
                time_indices = []

                for i, ts in enumerate(timestamps):
                    if ts < initial + offset * delta:
                        time_indices.append(i)
                    else:
                        split_indices.append(time_indices)
                        time_indices = []
                        offset += 1
                split_indices.append(time_indices)


            latent_vectors_list = [latent_vectors[split] for split in split_indices]
            unix_timestamps_list = [
                self.index_by_indices(unix_timestamps, split) for split in split_indices
            ]
            ymd_timestamps_list = [
                self.index_by_indices(ymd_timestamps, split) for split in split_indices
            ]
            post_ids_list = [
                self.index_by_indices(post_ids, split) for split in split_indices
            ]
            posts_list = [self.get_posts_from_post_ids(ids) for ids in post_ids_list]
            authors_list = [
                self.index_by_indices(authors, split) for split in split_indices
            ]

            vals_dict = {
                "data": self.make_data_dict(
                    latent_vectors_list,
                    unix_timestamps_list,
                    ymd_timestamps_list,
                    post_ids_list,
                    posts_list,
                    authors_list,
                ),
                "forums": forums,
                "msgs": msgs,
            }

        else:

            vals_dict = {
                "data": self.make_data_dict(
                    [latent_vectors],
                    [unix_timestamps],
                    [ymd_timestamps],
                    [post_ids],
                    [posts],
                    [authors],
                ),
                "forums": forums,
                "msgs": msgs,
            }

        return IronmarchAnalysis(
            dataset_path=self.dataset_path, dataset=self.dataset, values_dict=vals_dict
        )

    def make_data_dict(
        self, latent_vectors, unix_timestamps, ymd_timestamps, post_ids, posts, authors
    ):
        return {
            "latent_vectors": latent_vectors,
            "unix_timestamps": unix_timestamps,
            "ymd_timestamps": ymd_timestamps,
            "post_ids": post_ids,
            "posts": posts,
            "authors": authors,
        }

    def dists_to_ats(self):
        simplex_vertices = np.eye(self.latent_dim_size)
        dists_arr = []
        for latent in self.latent_vectors_list:
            dists = np.zeros((self.latent_dim_size, latent.shape[0]))
            for i, vertex in enumerate(simplex_vertices):
                dist = np.linalg.norm(latent - vertex, axis=1)
                dists[i] = dist
            dists_arr.append(dists)

        return dists_arr

    def make_violin_plot(self, save_to_folder: str = ''):
        dir = os.path.join(save_to_folder, "violin_plots")
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

        dists_arr = self.dists_to_ats()

        for idx, dists in enumerate(dists_arr):
            plt.figure(figsize=(8,5))
            plt.violinplot([dists[i] for i in range(self.latent_dim_size)], showmedians=True)
            plt.xticks(np.linspace(0, self.latent_dim_size, self.latent_dim_size+1))
            plt.xlabel('Archetypes')
            plt.ylabel('Distances')
            plt.title('{} to {}'.format(self.ymd_timestamps_list[idx][0], self.ymd_timestamps_list[idx][-1]))
            plt.savefig(os.path.join(dir, "violin_plot_{}.jpg".format(idx)))


    def get_nearest_indices(self, num_vectors_per_at: int) -> np.ndarray:
        """
        Returns array of indices closest to all archetypes.
        """

        # Computes distances from latent vectors to every corner of simplex
        nearest_indices = []
        nearest_vectors = []
        for latent in self.latent_vectors_list:
            dists = np.zeros((self.latent_dim_size, latent.shape[0]))
            for i, vertex in enumerate(np.eye(self.latent_dim_size)):
                dists[i] = np.sqrt(np.sum(np.square(latent - vertex), axis=1))

            # Get closest indices to each archetype by sorting
            indices = np.argsort(dists)[:, :num_vectors_per_at]
            nearest_indices.append(indices)

            vectors = np.zeros((self.latent_dim_size, num_vectors_per_at, self.latent_dim_size))
            for i, x in enumerate(indices):
                vectors[i] = latent[x]

            nearest_vectors.append(vectors)

        return nearest_indices, nearest_vectors

    # Gets specified number of archetypal posts
    # Only returns list of lists containing the posts. Update to have pandas option
    def get_archetypal_posts(self, num_posts_per_at: int, save_to_folder=""):
        all_at_posts = []
        all_at_timestamps = []
        all_at_authors = []

        nearest_indices, nearest_vectors = self.get_nearest_indices(num_posts_per_at)
        dir = os.path.join(save_to_folder, "archetypal_posts")
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

        for idx, (indices, vectors, posts, timestamps, authors, post_ids) in enumerate(
            zip(
                nearest_indices,
                nearest_vectors,
                self.posts_list,
                self.ymd_timestamps_list,
                self.authors_list,
                self.post_ids_list,
            )
        ):
            with open(os.path.join(dir, "top_archetypal_{}.txt".format(idx)), "w") as f:
                at_posts = []
                at_timestamps = []
                at_authors = []

                for i in range(self.latent_dim_size):
                    f.write("=" * 35 + " Archetype {} ".format(i) + "=" * 35)
                    f.write("\n")
                    top_posts = []
                    top_timestamps = []
                    top_authors = []

                    for j, k in enumerate(indices[i]):
                        f.write(
                            "{} -- {} -- Author: {} -- ID: {} -- Percentage: {:0.3f}\n{}".format(
                                j,
                                timestamps[k],
                                authors[k],
                                post_ids[k],
                                vectors[i][j][i],
                                re.sub("[\n\t\r]", " ", posts[k]),
                            )
                        )
                        f.write("\n\n")
                        top_posts.append(posts[k])
                        top_timestamps.append(timestamps[k])
                        top_authors.append(authors[k])

                    f.write("\n")

                    at_posts.append(top_posts)
                    at_timestamps.append(top_timestamps)
                    at_authors.append(top_authors)

                all_at_posts.append(at_posts)
                all_at_timestamps.append(at_timestamps)
                all_at_authors.append(at_authors)

        return {
            "posts": all_at_posts,
            "timestamps": at_timestamps,
            "authors": at_authors,
            "latent_vectors": nearest_vectors,
        }

    # Keyword extraction using TF-IDF algorithm. Return as pandas array
    # TODO: Provide a visualization that sorts TF-IDF scores for each archetype and makes a bar plot
    def archetypes_tfidf_scores(self, num_posts_per_at: int):
        # For each archetype, concatenate all posts into one string.
        # We will take all the posts in a single archetype to be one document
        nearest_indices, _ = self.get_nearest_indices(num_posts_per_at)
        all_at_posts = []
        for indices in nearest_indices:
            at_posts = []
            for i in range(self.latent_dim_size):
                top_posts = [self.posts[k] for k in indices[i]]
                post = ""
                for j in range(num_posts_per_at):
                    top = top_posts[j].lower()
                    top = re.sub(
                        r"<url>", " ", top
                    )  # Can probably combine these two re lines
                    top = re.sub("[\n\t\r]", " ", top)
                    post += top
                    post += ""
                at_posts.append(post)

            all_at_posts.append(at_posts)

        my_stop_words = text.ENGLISH_STOP_WORDS
        vectorizer = TfidfVectorizer(stop_words=my_stop_words)

        tfidf_dataframes = []
        for posts in at_posts:
            vectors = vectorizer.fit_transform(posts)
            feature_names = vectorizer.get_feature_names()
            dense = vectors.todense()
            denselist = dense.tolist()
            tfidf_dataframes.append(pd.DataFrame(denselist, columns=feature_names))

        return tfidf_dataframes

    # TODO: Some way to obtain weights and biases of side information linear head
    def get_weights_and_biases():
        pass

    # Visualizes pairwise angles between hyperplane normal vectors obtained from linear head
    def normal_hyperplanes(self, weights):
        class_names = [
            "nationality",
            "ethnicity",
            "religion",
            "gender",
            "sexual_orientation",
            "disability",
            "class",
        ]

        weights_norm = weights / np.linalg.norm(weights, axis=1)[:, np.newaxis]

        cos_similarity = np.zeros((7, 7))
        for i in range(7):
            for j in range(7):
                cos_similarity[i][j] = np.dot(weights_norm[i], weights_norm[j])

        sns.heatmap(
            cos_similarity,
            cmap=sns.cm.rocket_r,
            xticklabels=class_names,
            yticklabels=class_names,
        )

        closet_ats = []
        for normal in weights_norm:
            closet_ats.append(np.argmax(np.dot(np.eye(self.latent_dim_size), normal)))

        return cos_similarity, closet_ats

    # Get proportion of posts that lie on either side of hyperplane generated by side info
    def compute_hyperplane_proportions(self, weights, biases):
        proportions = []
        for i in range(7):
            count = 0
            for j in range(self.latent_vectors.shape[0]):
                if np.dot(weights[i], self.latent_vectors[j]) + biases[i] > 0:
                    count += 1
            proportions.append(count / self.latent_vectors.shape[0])

        print("Proportion of posts above each hyperplane")
        for idx, i in enumerate(class_names):
            print(class_names[idx] + ": " + str(proportions[idx]))

    # TODO: implement full dirichlet analysis of latent space
    def dirichlet_analysis(
        self,
        significance_level,
        dim,
        n_iter=10,
        sample_size=100,
        crit_val_arr=None,
        print_log=False,
    ):
        gof = DirichletGOF(significance_level, dim, sample_size, crit_val_arr)

        results = gof.test_statistic(
            self.latent_vectors, n_iter=n_iter, print_log=print_log
        )

        return results

    def __getitem__(self, key):
        return self.values_dict[key]

    def __str__(self):
        printout = 'Dictionary object with keys: "data", "forums", "msgs"\nValue of "data" is another dictionary with keys: "latent_vectors", "post_ids", "timestamps", "posts". Each key is a list containing {} element(s)'.format(
            len(self.values_dict["data"])
        )
        return printout

    def __repr__(self):
        printout = 'Dictionary object with keys: "data", "forums", "msgs"\nValue of "data" is another dictionary with keys: "latent_vectors", "post_ids", "timestamps", "posts". Each key is a list containing {} element(s)'.format(
            len(self.values_dict["data"])
        )
        return printout

    """
    track dirichlet-ness over time
    track centroid movement over time (all users/certain bad actors)
    use kernel density estimator to make plot along each archetypal direction (seaborn)
    csv of embedded coordinates, time, user id, location - maybe average each coordinate according to location
    """

    """
    [DONE] Get from author ids
    [DONE] Nearest posts per time period
    [DONE] cosine simililarity heat map
    [DONE] archetypes pointing towards
    have "get" return IronmarchAnalysis instance.
    rectangle - xaxis = time, yaxis = proportion of archetype
        each post has proportions of archetype
    """
