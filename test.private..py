import json 
import configparser

with open("pretrain.config.json", "r") as stream:

    out = json.load(stream)

    parser = configparser.ConfigParser()
    parser.read_dict(out)
    print(parser.model)

    
