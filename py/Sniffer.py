import scapy
import json
import numpy as np
import pandas as pd
from scapy.all import *
from seleniumwire import webdriver
from browsermobproxy import Server


def scappy_sniffer():
    sniffer = AsyncSniffer(filter='ip host 157.240.1.35')
    results = []

    for i in range(100):
        print(i)

        sniffer.start()

        options = webdriver.ChromeOptions()
        options.add_argument('headless')

        driver = webdriver.Chrome(chrome_options=options)
        driver.get('http://www.facebook.com')

        driver.close()
        sniffer.stop()

        results.append(sniffer.results)

    for i, result in enumerate(results):
        wrpcap(f'./pcap/session_{i}.pcapng', result)
