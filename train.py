import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hyperparam_file', type=str,
                  default='./hyperparam.txt',
                  help='hyperparameter file path')
parser.add_argument('--execute_file', type=str,
                  default='./train_net_pai.py',
                  help='the train file you want to execute')
parser.add_argument('--log_file', type=str,
                  default='./log.log',
                  help='log dir')
Flags,_ = parser.parse_known_args()

hyperparam_file = Flags.hyperparam_file
hyperparam = ""
with open(hyperparam_file, 'rt') as f:
	lines = f.readlines()
	for line in lines:
		if line == '\n': continue
		name = line.split('=')[0]
		content = line.split('=')[1][:-1]
		hyperparam += '--'+name+' '+content+' '

command = "python " + Flags.execute_file + " " + hyperparam
print(command)
stdout = os.popen(command)

