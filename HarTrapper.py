import configparser
import json, re, os
import pandas as pd
import requests
import selenium
from seleniumwire import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from browsermobproxy import Server


class HarTrapper:
    def __init__(self):
        # Setup settings from congig file.
        config = configparser.ConfigParser()
        config.read('config.ini')

        self.PAGE = config['CAPTURE_N_HAR_FILES']['PAGE']
        self.NAME = config['CAPTURE_N_HAR_FILES']['NAME']
        self.URL = config['CAPTURE_N_HAR_FILES']['URL']
        self.PATH = config['CAPTURE_N_HAR_FILES']['PATH']
        self.USERNAME = config['CAPTURE_N_HAR_FILES']['USERNAME']
        self.PASSWORD = config['CAPTURE_N_HAR_FILES']['PASSWORD']
        self.N = config['CAPTURE_N_HAR_FILES']['N']

        self.df = None

        self.server = Server(config['HAR']['SERVER_PATH'])
        self.server.start()
        self.proxy = self._start_proxy()
        self.driver = self._start_chrome_driver()


    def capture_n_har_files(self):
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

        print('Working...')

        # Enter facebook.com
        self.driver.get(self.URL)

        # Go to the 'pages' page in facebook.com
        self.go_to_pages()

        # Wait for 'c_user' cookie.
        while self.driver.get_cookie('c_user') is None:
            print('No c_user.')
            self.driver.implicitly_wait(2)

        # Click on the PAGE and close the tab N times.
        for i in range(int(self.N)):
            self.clear_cache()

            self.proxy.new_har(self.NAME)
            self.page_func()
            self.build_df()
            self.export_df(self.PATH + f'/_{i}.csv')

            print(f'{i}')

        # Finish session.
        self.driver.quit()
        self.server.stop()


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


    def clear_cache(self):
        cookies = self.driver.get_cookies()

        for cookie in cookies:
            name = cookie['name']

            if name == 'c_user' or name == 'xs':
                pass
            else:
                self.driver.delete_cookie(name)


    def export_har(self):
        """Export the har_fit recording to a json file.
        """
        with open('./har_fit.json', 'w') as file:
            json.dump(self.proxy.har, file)


    def export_df(self, path):
        """Export the instance DataFrame to a csv file.
        :param path: Export directory path.
        """
        self.df.to_csv(path)


    def go_to_pages(self):
        timeout = 5

        email_xpath = '//input[@id="email"] | //input[@name="email"]'
        pass_xpath = '//input[@id="pass"] | //input[@name="pass"]'
        login_xpath = '//input[@value="Log In"] | //button[@name="login"]'

        run = True

        while run:
            try:
                WebDriverWait(self.driver, timeout).until(EC.presence_of_element_located((By.XPATH, email_xpath)))
                WebDriverWait(self.driver, timeout).until(EC.presence_of_element_located((By.XPATH, login_xpath)))
                WebDriverWait(self.driver, timeout).until(EC.presence_of_element_located((By.XPATH, pass_xpath)))
                run = False

            except selenium.common.exceptions.NoSuchElementException:
                print('Login NoSuchElementException.')
            except selenium.common.exceptions.TimeoutException:
                print('Login TimeoutException.')
            except selenium.common.exceptions.ElementNotInteractableException:
                print('Login ElementNotInteractableException.')

        self.driver.find_element_by_xpath(email_xpath).send_keys(self.USERNAME)
        self.driver.find_element_by_xpath(pass_xpath).send_keys(self.PASSWORD)
        self.driver.find_element_by_xpath(login_xpath).click()

        run = True

        while run:
            try:
                WebDriverWait(self.driver, timeout).until(
                    EC.presence_of_element_located((By.XPATH, '//div[text()="Pages"]')))
                self.driver.find_element_by_xpath('//div[text()="Pages"]').click()
                run = False

            except selenium.common.exceptions.NoSuchElementException:
                print('Login NoSuchElementException.')
            except selenium.common.exceptions.TimeoutException:
                print('Login TimeoutException.')
            except selenium.common.exceptions.ElementNotInteractableException:
                print('ElementNotInteractableException')


    def page_func(self):
        """Passed to the Har.capture_n_har_files procedure for selenium to run
        on the web page.
        """
        timeout = 5

        page_xpath = '//span[text()="' + self.PAGE + '"]'
        home_xpath = '//a[text()="Home"]'

        # Make sure all elements exist on page before moving on.
        pages_window = self.driver.current_window_handle

        run = True

        while run:
            try:
                WebDriverWait(self.driver, timeout).until(EC.presence_of_element_located((By.XPATH, page_xpath)))
                self.driver.find_element_by_xpath(page_xpath).click()
                run = False

            except selenium.common.exceptions.NoSuchElementException:
                print('Login NoSuchElementException.')
            except selenium.common.exceptions.TimeoutException:
                print('Login TimeoutException.')
            except selenium.common.exceptions.ElementNotInteractableException:
                print('ElementNotInteractableException')

        run = True

        while run:
            try:
                new_window = self.driver.window_handles[1]
                self.driver.switch_to_window(new_window)
                self.driver.close()
                self.driver.switch_to_window(pages_window)

                run = False

            except selenium.common.exceptions.NoSuchElementException:
                print('Login NoSuchElementException.')
            except selenium.common.exceptions.TimeoutException:
                print('Login TimeoutException.')
            except selenium.common.exceptions.ElementNotInteractableException:
                print('Login ElementNotInteractableException.')


    def build_df(self):
        """Iterate each row in the har_fit data csv file
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


har_trapper = HarTrapper()
har_trapper.capture_n_har_files()
