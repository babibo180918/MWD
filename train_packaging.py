import os
import argparse
import yaml

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

parser = argparse.ArgumentParser(prog="packaging")
parser.add_argument("--config", "-c", type=str, default=DEFAULT_YAML, help="The file path of configuration file")
parser.add_argument("--epochs", type=int,required=True ,default = 10, help="number of epochs")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate for the optimizer")

args = parser.parse_args()

print(args)

with open(args.config) as config_file:
    config = yaml.safe_load(config_file)
    print(config)
    print(config["learning"]["running"]["num_epochs"])
