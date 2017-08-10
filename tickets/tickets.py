# coding: utf-8
"""命令行火车票查看器

Usage:
	tickets [-gdtkz] <from> <to> <date>

Options:
	-h,--help 显示帮助菜单
	-g        高铁
	-d        动车
	-t        特快
	-k        快速
	-z        直达

Example:
	tickets 北京 上海 2016-10-10
	tickets -dg 成都 南京 2016-10-10
"""
from docopt import docopt
from stations_name import stations
import requests
from prettytable import PrettyTable
from colorama import init,Fore
init()
#requests.packages.urllib3.disable_warnings()
class TrainCollection:
    header = '车次 车站 时间 历时 商务 一等 二等 软卧 硬卧 硬座 无座'.split()
    def __init__(self,trains,options):
        self.trains = trains
        self.options = options
        self.code_dict = {v:k for k,v in stations.items()}
    def get_info(self):
        for train in self.trains:
            data_list = train.split('|')
            train_num = data_list[3]
            if len(self.options)!=0 and (not ('-'+str(train_num[0].lower()) in self.options)):
                continue
            from_station_code = data_list[6]
            from_station_name = self.code_dict[from_station_code]
            to_station_code = data_list[7]
            to_station_name = self.code_dict[to_station_code]
            start_time = data_list[8]
            arrive_time = data_list[9]
            run_time = data_list[10]
            special_seat = data_list[32] or '--'
            first_seat = data_list[31] or '--'
            second_seat = data_list[30] or '--'
            soft_sleep = data_list[23] or '--'
            hard_sleep = data_list[28] or '--'
            hard_seat = data_list[29] or '--'
            no_seat = data_list[26] or '--'
            train = [train_num,
                     '\n'.join([Fore.GREEN + from_station_name + Fore.RESET,
                                Fore.LIGHTRED_EX + to_station_name + Fore.RESET]),
                     '\n'.join([Fore.GREEN + start_time + Fore.RESET,
                                Fore.LIGHTRED_EX+ arrive_time + Fore.RESET]),
                     run_time.replace(':','小时')+'分',
                     special_seat,
                     first_seat,
                     second_seat,
                     soft_sleep,
                     hard_sleep,
                     hard_seat,
                     no_seat
                    ]
            yield train
    def pretty_print(self):
        pt = PrettyTable()
        pt._set_field_names(self.header)
        for train in self.get_info():
            pt.add_row(train)
        print(pt)
def cli():
    arguments = docopt(__doc__)
    #print(arguments)
    from_station = stations.get(str(arguments['<from>']))
    to_starion = stations.get(str(arguments['<to>']))
    date = arguments['<date>']
    #print (from_station,to_starion,date)
    url = 'https://kyfw.12306.cn/otn/leftTicket/query?leftTicketDTO.train_date={}&leftTicketDTO.from_station={}&leftTicketDTO.to_station={}&purpose_codes=ADULT'.format(
        date,from_station,to_starion
    )
    #print (url)
    r = requests.get(url,verify=False)
    options =([key for key,value in arguments.items() if value is True])
    try:
        TrainCollection((r.json()['data']['result']),options).pretty_print()
    except:
        try:
            print (r.json()['messages'])
        except:
            print ('Input Error')
if __name__ == '__main__':
    cli()

