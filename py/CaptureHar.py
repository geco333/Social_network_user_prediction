import json, re, os, random, logging, time, traceback
from functools import reduce
import pandas as pd
from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt
import requests
import selenium
from seleniumwire import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from browsermobproxy import Server
from collections import Counter
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction import DictVectorizer


class Har:
    @staticmethod
    def from_csv(path: str) -> list:
        """Transform each har_fit file in the path to a Pandas DataFrame,
            add each DataFrame to a list,
            and return the list.

        :param str path: The path to the har_fit files directory.
        :return list: A list of the har_fit files data in Pandas DataFrames.
        """

        hars = []

        for file in os.listdir(path):
            if file[-4:] == '.csv':
                hars.append(pd.read_csv(f'{path}/{file}'))

        return hars


    @staticmethod
    def capture_n_har_files(path: str, page: str = None, n: int = 1, name: str = '', url: str = '', rnd: bool = False,
                            page_func=''):
        """Run n times:
            Create an Har class instance.
            Using BrowsermobProxy start recording har data from the browser.
            Using selenium start a browser and go to the url.
            Perform the action ordered by the 'page_func' function.
            Create a Pandas DataFrame from the har data recorded.
            Close the selenium session and the proxy.
            Export the DataFrame to a csv file.

        :param page_func: A custom function for action to perform inside the webpage.
        :param path: csv file/s save location.
        :param rnd: Whether to choose a random url.
        :param n: Number of times to run.
        :param name: Record name.
        :param url: The web site, use full address(exm: http://www.google.com).
        """

        urls = ['https://www.tumblr.com/', 'https://findtheinvisiblecow.com/', 'https://theuselessweb.com/',
                'https://www.linkedin.com/', 'https://www.reddit.com/', 'https://www.taringa.net/',
                'https://the-dots.com/', 'https://www.youtube.com/',
                'https://www.reverbnation.com/', 'https://www.flixster.com/', 'https://www.care2.com/',
                'https://www.ravelry.com/account/login'
                'http://hackertyper.com/', 'https://www.instagram.com/', 'https://twitter.com/',
                'https://www.pinterest.com/']

        try:
            _ = re.findall('\\d+', os.listdir(f'{path}').__str__())
            __ = list(map(lambda x: int(x), _))
            last_i = max(__) + 1
        except ValueError:
            last_i = 0

        for i in range(last_i, last_i + n):
            if rnd:
                url = urls[random.randrange(len(urls))]

            print(str(i) + ' ' + url)

            har = Har()
            har._capture_data(name, url, page_func=page_func, page=page)
            har._build_df()
            har.quit()

            har.export_df(f'{path}/har_df_{i}.csv')


    def _capture_data(self, name, url, page_func, page):
        """
        :param name: The har output name.
        :param url: The website to capture har data from.
        :param page_func: A custom function ordering the actions to perform inside the web page.
        """

        self.proxy.new_har(name)
        self.driver.get(url)

        if page_func != '':
            page_func(self.driver, page)


    def __init__(self, path=None):
        if path is None:
            self.server = Server(
                'C:/Users/Geco/AngularProjects/BuildUrlDatabase/py/browsermob-proxy-2.1.4/bin/browsermob-proxy.bat')
            self.server.start()
            self.proxy = self._start_proxy()
            self.driver = self._start_chrome_driver()
            self.df = None
        else:
            self.df = pd.read_csv(f'{path}')


    def _start_proxy(self):
        """Start a new proxy server to capture har data.

        :return: The new server proxy.
        """

        run = True

        while run:
            try:
                proxy = self.server.create_proxy()
                run = False
            except requests.exceptions.ConnectionError as e:
                print(e)

        return proxy


    def _start_chrome_driver(self) -> webdriver:
        """Using Selenium start the google chrome browser headless.
        All the browser requests and responses(har_fit data) will be recorded
        using a BrowsermobProxy proxy server.

        :return: Google chrome driver object.
        """

        chrome_options = webdriver.ChromeOptions()
        prefs = {"profile.default_content_setting_values.notifications": 2}
        chrome_options.add_experimental_option("prefs", prefs)
        chrome_options.set_capability('proxy', {'httpProxy': f'{self.proxy.proxy}',
                                                'noProxy': '',
                                                'proxyType': 'manual',
                                                'sslProxy': f'{self.proxy.proxy}'})
        # chrome_options.add_argument("--headless")

        driver = webdriver.Chrome(chrome_options=chrome_options)
        driver.set_page_load_timeout(999)
        driver.delete_all_cookies()

        return driver


    def quit(self):
        """Close all open connections: Close the proxy server and the chrome driver.
        """
        self.driver.quit()
        self.server.stop()
        os.system("taskkill /f /im java.exe")


    def export_har(self):
        """
        Export the har_fit recording to a json file.
        """
        with open('./har_fit.json', 'w') as file:
            json.dump(self.proxy.har, file)


    def export_df(self, path):
        """
        Export the instance DataFrame to a csv file.
        :param path: Export directory path.
        """
        self.df.to_csv(path)


    def _add_to_dict(self, __, k, v):
        """Utility method for the build_df method.
        """
        if type(v) == list:
            for kk, vv in v:
                if type(vv) == dict or type(vv) == list:
                    self._add_to_dict(__, k + kk + '.', vv)
                else:
                    __[k + kk] = vv
        else:
            for kk, vv in v.items():
                if type(vv) == dict or type(vv) == list:
                    self._add_to_dict(__, k + kk + '.', vv)
                else:
                    __[k + kk] = vv


    def _build_df(self):
        """
        Iterate each row in the har_fit data csv file
        and add it to a dictionary.
        Add all the rows dictionaries to a list.
        Create one complete DataFrame from the list.

        :return: The instance har_fit recording data in the form of a Pandas DataFrame.
        """
        _ = list()

        for entry in self.proxy.har['log']['entries']:
            __ = dict()

            for k, v in entry.items():
                if type(v) == dict or type(v) == list:
                    self._add_to_dict(__, k + '.', v)
                else:
                    __[k] = v

            _.append(__)

        self.df = pd.DataFrame(_)


class FingerPrint:
    def __init__(self, hars, types: bool = False):
        self.hars = hars
        self.length = len(hars)
        self.sums = []
        self.sessions = []
        self.weights = None

        self._init_data()
        self._init_weights()

        if types:
            self.types_counts = self._gather_types()
            self.types = self._get_types()


    def _gather_types(self):
        _ = []

        for df in self.hars:
            __ = {}

            for row in df.iterrows():
                try:
                    ___ = re.findall('(?<=\.)\w{1,4}$', row[1]['request.url'])[0]

                    if ___ in __:
                        __[___] += 1
                    else:
                        __[___] = 1
                except IndexError:
                    pass

            _.append(__)

        return _


    def _get_types(self):
        _ = []

        for session in self.types_counts:
            for k in session.keys():
                if k not in _:
                    _.append(k)

        return _


    def _init_data(self):
        for har in self.hars:
            session_sums = [0, 0]
            session = []

            for row in har[['response.bodySize', 'response.headersSize', 'time']].values:
                row = tuple(row.tolist())

                # Add to sums.
                session_sums[0] += row[0]
                session_sums[1] += row[1]

                session.append(row)

            # Add to count.
            self.sessions.append(session)
            self.sums.append(session_sums)


    def _flatten_sessions(self):
        return [row for session in self.sessions for row in session]


    def _init_weights(self):
        flat_sessions = self._flatten_sessions()
        self.weights = Counter(flat_sessions)


class ResponseData:
    def __init__(self, hars: list, types: bool = False):
        self.hars = hars
        self.length = len(hars)
        self.sums = []
        self.sessions = []

        self._init_data()

        if types:
            self.types_counts = self._gather_types()


    def _init_data(self):
        """Sum body size and header size individually for each session: (body size, header size)
           Create a tuple of (body size, header size) per session then add each session to self.sessions."""
        for har in self.hars:
            session_sums = [0, 0]
            session = []

            for row in har[['response.bodySize', 'response.headersSize', 'time']].values:
                row = tuple(row.tolist())

                session_sums[0] += row[0]
                session_sums[1] += row[1]

                session.append(row)

            self.sums.append(session_sums)
            self.sessions.append(session)


    def _gather_types(self):
        _ = []

        for df in self.hars:
            __ = {}

            for row in df.iterrows():
                try:
                    ___ = re.findall('(?<=\.)\w{1,4}$', row[1]['request.url'])[0]

                    if ___ in __:
                        __[___] += 1
                    else:
                        __[___] = 1
                except IndexError:
                    pass

            _.append(__)

        return _


class Analyzer:
    def __init__(self, fp: FingerPrint, rd: ResponseData):
        self.fp = fp  # Fit data.
        self.rd = rd  # Predict data.
        self.x_fit = []
        self.x_predict = []
        self.y_fit_true = []
        self.y_predict_true = []


    def plot_confusion_matrix(self, title=None, cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        # Compute confusion matrix
        cm = confusion_matrix(self.y_predict_true, self.predictions)
        # Only use the labels that appear in the data
        classes = unique_labels(self.y_fit_true, self.y_predict_true)

        print(classification_report(self.y_predict_true, self.predictions, labels=[0, 1],
                                    target_names=['facebook', 'other']))

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        thresh = cm.max() / 2.

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()

        return ax


class TypesAnalyzer(Analyzer):
    def __init__(self, fp: FingerPrint, rd: ResponseData):
        super().__init__(fp, rd)

        self.dict_vectorizer = DictVectorizer(sparse=False)
        self._init_data()
        self.clf = self._classify()
        self.predictions = self.clf.predict(self.x_predict)


    def _init_data(self):
        _ = []

        for session_types in self.rd.types_counts:
            __ = {}

            for t in self.fp.types:
                if t in session_types:
                    __[t] = session_types[t]
                else:
                    __[t] = 0

            _.append(__)

        self.x_fit = self.dict_vectorizer.fit_transform(self.fp.types_counts)
        self.y_fit_true = [1] * self.x_fit.shape[0]
        self.x_predict = self.dict_vectorizer.transform(_)
        self.y_predict_true = [0] * self.x_predict.shape[0]


    def _classify(self):
        self.x_fit, self.x_predict, self.y_fit_true, self.y_predict_true = train_test_split(
            np.concatenate([self.x_fit.data, self.x_predict.data]),
            np.concatenate([self.y_fit_true, self.y_predict_true]),
            test_size=0.9)

        return svm.SVC(gamma='scale', kernel='rbf').fit(self.x_fit, self.y_fit_true)


class WeightsAnalyzer(Analyzer):
    def __init__(self, fp: FingerPrint, rd: ResponseData):
        super().__init__(fp, rd)

        self._init_data()
        self.clf = self._classify()
        self.predictions = self.clf.predict(self.x_predict)

        self.score = cross_val_score(self.clf, self.x_fit, y=self.y_fit_true, cv=5, scoring='f1_weighted')


    def _score_sessions(self, sessions: list, label: int):
        _ = []
        __ = []

        for session in sessions:
            session_score = 0

            for feature in self.fp.weights.keys():
                if feature in session:
                    session_score += 1

            _.append([session_score])
            __.append(label)

        return np.array(_), np.array(__)


    def _init_data(self):
        self.x_fit, self.y_fit_true = self._score_sessions(self.fp.sessions, 1)
        self.x_predict, self.y_predict_true = self._score_sessions(self.rd.sessions, 0)


    def _classify(self):
        self.x_fit, self.x_predict, self.y_fit_true, self.y_predict_true = train_test_split(
            np.concatenate([self.x_fit, self.x_predict]), np.concatenate([self.y_fit_true, self.y_predict_true]),
            test_size=0.9)

        return svm.SVC(gamma='scale', kernel='rbf').fit(self.x_fit, self.y_fit_true)


class SumsAnalyzer(Analyzer):
    def __init__(self, fp: FingerPrint, rd: ResponseData):
        super().__init__(fp, rd)

        self._init_data()
        self.clf = self._classify()
        self.predictions = self.clf.predict(self.x_predict)

        self.score = cross_val_score(self.clf, self.x_fit, y=self.y_fit_true, cv=5, scoring='f1_weighted')


    def _init_data(self):
        length = min(len(self.fp.hars), len(self.rd.hars))

        self.x_fit, self.x_predict, self.y_fit_true, self.y_predict_true = train_test_split(
            np.concatenate([self.fp.sums[:length], self.rd.sums[:length]]), [1] * length + [0] * length, test_size=0.9)


    def _classify(self):
        return svm.SVC(gamma='scale', kernel='rbf').fit(self.x_fit, self.y_fit_true)


    def plot_scatter(self):
        x_fp = np.array(self.fp.sums)[:, 0]
        y_fp = np.array(self.fp.sums)[:, 1]

        x_rd = np.array(self.rd.sums)[:, 0]
        y_rd = np.array(self.rd.sums)[:, 1]

        plt.scatter(x_rd, y_rd, label='other')
        plt.scatter(x_fp, y_fp, label='facebook')
        plt.legend()
        plt.ylabel('Size')
        plt.xlabel('Time')


class CombinedAnalyzer(Analyzer):
    def __init__(self, fp: FingerPrint, rd: ResponseData):
        super().__init__(fp, rd)

        self.dict_vectorizer = DictVectorizer(sparse=False)
        self._init_data()
        self.clf = self._classify()
        self.predictions = self.clf.predict(self.x_predict)

        self.score = cross_val_score(self.clf, self.x_fit, y=self.y_fit_true, cv=5, scoring='f1_weighted')


    def _init_data(self):
        length = min(self.fp.length, self.rd.length)

        # Add the type feature.
        self.x_fit = self.dict_vectorizer.fit_transform(self.fp.types_counts[:length]).tolist()
        self.x_predict = self.dict_vectorizer.transform(self.rd.types_counts[:length]).tolist()

        for session_i in range(length):
            # Add the sums feature.
            self._add_sums(session_i)

            # Add the weight score. 0 by default.
            self.x_fit[session_i].append(0)
            self.x_predict[session_i].append(0)

            # Add the weights feature.
            self._add_weights(session_i)

            # Add labels
            self.y_fit_true.append(1)
            self.y_predict_true.append(0)


    def _add_sums(self, i):
        self.x_fit[i].append(self.fp.sums[i][0])
        self.x_fit[i].append(self.fp.sums[i][1])
        self.x_predict[i].append(self.rd.sums[i][0])
        self.x_predict[i].append(self.rd.sums[i][1])


    def _add_weights(self, i):
        for feature in self.fp.weights.keys():
            if feature in self.fp.sessions[i]:
                self.x_fit[i][-1] += 1
            if feature in self.rd.sessions[i]:
                self.x_predict[i][-1] += 1


    def _classify(self):
        self.x_fit, self.x_predict, self.y_fit_true, self.y_predict_true = train_test_split(
            np.concatenate([self.x_fit, self.x_predict]), np.concatenate([self.y_fit_true, self.y_predict_true]),
            test_size=0.9)

        return svm.SVC(gamma='scale', kernel='rbf').fit(self.x_fit, self.y_fit_true)


class UsersAnalyzer:
    def __init__(self, data: list, flags: str = 't'):
        self.flags = flags
        self.data = data
        self.dict_vectorizer = DictVectorizer(sparse=False)
        self.x_fit = []
        self.y_fit_true = []

        self._init()
        self.clf = self._classify()
        self.predictions = self.clf.predict(self.x_predict)
        self.score = cross_val_score(self.clf, self.x_fit, y=self.y_fit_true, cv=5)


    def _init(self):
        if 't' in self.flags:
            _ = reduce(lambda a, b: a + b, [x.types_counts for x in self.data])
            self.x_fit = self.dict_vectorizer.fit_transform(_)
        elif 's' in self.flags:
            self.x_fit = np.array(reduce(lambda a, b: a + b, [x.sums for x in self.data]))
        elif 'w' in self.flags:
            self.x_fit = np.array(reduce(lambda a, b: a + b, [x.weights for x in self.data]))

        self.y_fit_true = np.array(reduce(lambda a, b: a + b,
                                          [[i] * len(self.data[i].types_counts) for i in range(len(self.data))]))


    def _classify(self):
        sss = StratifiedShuffleSplit(n_splits=len(self.data), test_size=0.5)
        _, __ = [], []

        for fit_i, train_i in sss.split(self.x_fit, self.y_fit_true):
            _.extend(self.x_fit[fit_i])
            self.x_predict = self.x_fit[train_i]
            __.extend(self.y_fit_true[fit_i])
            self.y_predict_true = self.y_fit_true[train_i]

        self.x_fit = _
        self.y_fit_true = __

        return svm.SVC(gamma='scale', kernel='rbf').fit(self.x_fit, self.y_fit_true)


    def plot_confusion_matrix(self, title=None, cmap=plt.cm.Blues):
        """This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        # Compute confusion matrix
        cm = confusion_matrix(self.y_predict_true, self.predictions)
        # Only use the labels that appear in the data
        classes = unique_labels(self.y_fit_true, self.y_predict_true)

        print(classification_report(self.y_predict_true, self.predictions, labels=range(len(self.data)),
                                    target_names=[str(i) for i in range(len(self.data))]))

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        thresh = cm.max() / 2.

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()


def page_func(driver, page):
    """Passed to the Har.capture_n_har_files procedure for selenium to run
    on the web page.
    """
    timeout = 10

    email_xpath = '//input[@id="email"] | //input[@name="email"]'
    pass_xpath = '//input[@id="pass"] | //input[@name="pass"]'
    login_xpath = '//input[@value="Log In"] | //button[@name="login"]'
    page_xpath = '//span[text()="' + page + '"]'

    # Make sure all elements exist on page before moving on.
    run = True

    while run:
        try:
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, email_xpath)))
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, login_xpath)))
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, pass_xpath)))

            run = False
        except selenium.common.exceptions.NoSuchElementException:
            print('Login NoSuchElementException.')
        except selenium.common.exceptions.TimeoutException:
            print('Login TimeoutException.')

    driver.find_element_by_xpath(email_xpath).send_keys('gggppp282@gmail.com')
    driver.find_element_by_xpath(pass_xpath).send_keys('g31012310G')
    driver.find_element_by_xpath(login_xpath).click()

    # Make sure all elements exist on page before moving on.
    run = True

    while run:
        try:
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, '//div[text()="Pages"]')))

            run = False
        except selenium.common.exceptions.NoSuchElementException:
            print('Login NoSuchElementException.')
        except selenium.common.exceptions.TimeoutException:
            print('Login TimeoutException.')

    driver.find_element_by_xpath('//div[text()="Pages"]').click()

    # Make sure all elements exist on page before moving on.
    run = True

    while run:
        try:
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, page_xpath)))

            run = False
        except selenium.common.exceptions.NoSuchElementException:
            print('Login NoSuchElementException.')
        except selenium.common.exceptions.TimeoutException:
            print('Login TimeoutException.')

    driver.find_element_by_xpath(page_xpath).click()


def capture_har_data(n, page_func=None, page=None):
    """Utility procedure to create n HAR files, both for the fingerprint
        and random.
    """
    for i in range(n):
        print('\nCapturing FingerPrint HAR data...')
        Har.capture_n_har_files(path='./har_fit', n=1, url='https://www.facebook.com', name='facebook',
                                page_func=page_func, page=page)

        # print('\nCapturing ResponseData HAR data...')
        # Har.capture_n_har_files(path='./har_random', n=1, rnd=True)


def run_analyzers(fp, rd):
    wa = WeightsAnalyzer(fp, rd)
    sa = SumsAnalyzer(fp, rd)
    ta = TypesAnalyzer(fp, rd)
    ca = CombinedAnalyzer(fp, rd)

    wa.plot_confusion_matrix(title='WeightsAnalyzer')
    sa.plot_confusion_matrix(title='SumsAnalyzer')
    ta.plot_confusion_matrix(title='TypesAnalyzer')
    ca.plot_confusion_matrix(title='CombinedAnalyzer')


users = ['./har_fit_0', './har_fit_1', './har_fit_2', './har_fit_3']
fp = [FingerPrint(Har.from_csv(user), types=True) for user in users]
ua = UsersAnalyzer(fp, flags='t')
