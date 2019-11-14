import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from scapy.all import *
from collections import Counter
from functools import reduce


class PcapDataFrame:
    @staticmethod
    def plot_sequences(pcaps) -> 'Plot the dataframes: x=Time, y=sequence number':
        width = 3
        cols = math.ceil((len(pcaps) / 3))
        rows = len(pcaps)
        fig, axs = plt.subplots(cols, rows, sharey=True)

        for i, pcap in enumerate(pcaps):
            x = np.arange(len(pcap.data))
            _x = np.arange(len(pcap.data) / 2)
            requests = [pcap.data.loc[i]['size'] for i in x if pcap.data.loc[i]['type'] == 'response']
            responses = [pcap.data.loc[i]['size'] for i in x if pcap.data.loc[i]['type'] == 'request']

            axs[i].bar(_x, requests, width=width)
            axs[i].bar(_x, responses, width=width)
            axs[i].set_title(pcap.path)


    @staticmethod
    def plot_response_sizes(pcaps: list) -> 'Plot the sizes_count DataFrame: x=sizes, y=size count':
        request_sums = [pcap.data[pcap.data['type'] == 'request']['size'].sum() for pcap in pcaps]
        response_sums = [pcap.data[pcap.data['type'] == 'response']['size'].sum() for pcap in pcaps]
        request_and_response_sums = [pcap.data['size'].sum() for pcap in pcaps]
        response_times = [float(pcap.data[pcap.data['type'] == 'response']['time_elapsed'].sum()) for pcap in pcaps]

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].hist(request_sums)
        axs[0, 0].set_title('request_sums')
        axs[0, 1].hist(response_sums)
        axs[0, 1].set_title('response_sums')
        axs[1, 0].hist(request_and_response_sums)
        axs[1, 0].set_title('request_and_response_sums')
        axs[1, 1].hist(response_times)
        axs[1, 1].set_title('response_times')


    @staticmethod
    def plot_total_sizes(pcaps):
        sums = [pcap.data['size'].sum() for pcap in pcaps]
        average = np.average(sums)
        x = [str(sum) for sum in sums]

        fig, axs = plt.subplots(1, 2)
        axs[0].hist(sums)
        axs[1].bar(np.arange(len(x)), sums)
        axs[1].plot(x, [average] * len(x), c='r')


    def __init__(self, path):
        self.path = path  # Raw data(pcapng file) path.
        self.pcap = rdpcap(path)  # Scapy output.
        self.df = pd.DataFrame()  # The scapy output in a pandas DataFrame.
        self.data = pd.DataFrame()  # Request-response data.
        self.sizes_count = pd.DataFrame()  # Counts the number of times each unique size appears in the data.

        self.init_data()
        self.get_packets()
        self.count_sizes()


    def init_data(self):
        __ = list()

        try:
            real_request_seq = self.pcap[0]['TCP'].seq
            real_response_seq = self.pcap[1]['TCP'].seq
            self_mac = self.pcap[0]['Ether'].src
        except:
            print(self.path)

        for packet in self.pcap:
            _ = dict()

            _['timestamp'] = packet.time - self.pcap[0].time

            if packet.haslayer(scapy.layers.l2.Ether):
                for field in list(packet['Ether'].fields.items()):
                    if field[0] == 'src':
                        if field[1] == 'e0:94:67:c5:af:21':
                            _['src'] = 'Client host'
                            _['dst'] = 'facebook.com'
                        else:
                            _['src'] = 'facebook.com'
                            _['dst'] = 'Client host'
                    else:
                        _['Ether.' + field[0]] = field[1]

            if packet.haslayer(scapy.layers.inet.IP):
                for field in list(packet['IP'].fields.items()):
                    if field[0] == 'options':
                        for option in field[1]:
                            _['IP.options.' + option[0]] = option[1]
                    elif field[0] == 'flags':
                        _['IP.' + field[0]] = field[1].flagrepr()
                    else:
                        _['IP.' + field[0]] = field[1]

            if packet.haslayer(scapy.layers.inet.TCP):
                for field in list(packet['TCP'].fields.items()):
                    if field[0] == 'options':
                        for option in field[1]:
                            _['TCP.options.' + option[0]] = option[1]
                    elif field[0] == 'flags':
                        _['TCP.' + field[0]] = field[1].flagrepr()
                    elif field[0] == 'seq':
                        if packet['Ether'].src == self_mac:
                            _['TCP.' + field[0]] = field[1] - real_request_seq
                        else:
                            _['TCP.' + field[0]] = field[1] - real_response_seq
                    elif field[0] == 'ack':
                        if packet['Ether'].src == self_mac:
                            if 'S' in packet['TCP'].flags.flagrepr():
                                _['TCP.' + field[0]] = 0
                            else:
                                _['TCP.' + field[0]] = field[1] - real_response_seq
                        else:
                            _['TCP.' + field[0]] = field[1] - real_request_seq
                    else:
                        _['TCP.' + field[0]] = field[1]

            if packet.haslayer(scapy.packet.Raw):
                _['Raw.load'] = packet['Raw'].load
                _['Raw.load.size'] = len(packet['Raw'].load)
            else:
                _['Raw.load.size'] = 0

            __.append(_)

        self.df = pd.DataFrame(__).sort_values('timestamp')


    def get_packets(self):
        df_gen = self.df.iterrows()
        last_row = pd.Series()
        size = 0
        id = 0
        load = bytes()
        __ = list()

        for i, row in df_gen:
            if 'dst' in last_row:
                size += row['Raw.load.size']
                load += row['Raw.load'] if 'Raw' in row else bytes()

                if row['src'] != last_row['src']:
                    _ = dict()

                    if last_row['dst'] == 'facebook.com':
                        _['type'] = 'request'
                        _['load'] = load
                        _['time_elapsed'] = ''
                        _['size'] = size
                        _['id'] = id
                        _['associated_id'] = ''
                        __.append(_)
                    elif last_row['dst'] == 'Client host':
                        _['type'] = 'response'
                        _['load'] = load
                        _['time_elapsed'] = row['timestamp'] - last_row['timestamp']
                        _['size'] = size
                        _['id'] = ''
                        _['associated_id'] = id
                        id += 1
                        __.append(_)

                    size = row['Raw.load.size']
                    load = row['Raw.load'] if type(row['Raw.load']) is bytes else bytes()

            last_row = row

        self.data = pd.DataFrame(__)


    def show(self, columns=None) -> 'Prints the DataFrame':
        '''Print the DataFrame with the columns specified in the columns parameter.

        param columns: A list containing one or more of the following:
           'Ether.dst'
           'Ether.type'
           'IP.chksum'
           'IP.dst'
           'IP.flags'
           'IP.frag'
           'IP.id'
           'IP.ihl'
           'IP.len'
           'IP.proto'
           'IP.src'
           'IP.tos'
           'IP.ttl',
           'IP.version'
           'Raw.load'
           'Raw.load.size'
           'TCP.ack'
           'TCP.chksum'
           'TCP.dataofs'
           'TCP.dport'
           'TCP.flags'
           'TCP.options.MSS'
           'TCP.options.NOP'
           'TCP.options.SAckOK'
           'TCP.options.WScale'
           'TCP.reserved'
           'TCP.seq'
           'TCP.sport'
           'TCP.urgptr'
           'TCP.window'
           'dst'
           'src'
           'timestamp'
        '''
        columns = ['timestamp', 'src', 'dst', 'TCP.seq', 'TCP.ack', 'Raw.load.size'] if columns.eq(None) else columns
        print(self.df[columns])


    def export_data(self):
        self.data.to_csv(path_or_buf='./data.csv', encoding='utf-8')


    def count_sizes(self):
        try:
            counter = Counter(self.data['size'])
            self.sizes_count = pd.DataFrame(counter.values(), index=counter.keys()).sort_index()
        except:
            print(self.path)
