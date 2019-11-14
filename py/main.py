from PcapDataFrame import PcapDataFrame
import matplotlib.pyplot as plt
import pandas as pd
import os


pcaps = []

for file in os.listdir('./pcap'):
    if file[-7:] == '.pcapng':
        pcaps.append(PcapDataFrame('./pcap/' + file))

PcapDataFrame.plot_response_sizes(pcaps=pcaps)

# ds = pd.concat([pcap.data for pcap in pcaps], ignore_index=True)
# ds.to_csv(path_or_buf='./ds.csv', encoding='utf-8')
