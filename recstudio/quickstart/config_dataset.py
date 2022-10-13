import os
import yaml
from ..utils.utils import set_color, get_dataset_default_config


def type_promt_rec(types, i):
    l = []
    for j, t in enumerate(types):
        if j == i:
            l.append(f"{j}. {t} (Recommendation)")
        else:
            l.append(f"{j}. {t}")
    return "\n".join(l)

def print_promt(s):
    print(set_color(s, 'green'))

def print_opt():
    promt = set_color(">>> ", 'cyan')
    return input(f"{promt}")

def _add_feat(dir, feat, file_promt, files, optional=False):
    promt_str = f"Please select the {feat} file (csv type file and header is required.)"
    if optional:
        promt_str += "(Optional)"
    print_promt(f"{promt_str}:")
    print(f"{file_promt}")
    option = print_opt()

    if (len(option) == 0) and optional:
        return None
    else:
        feat_file = files[int(option)]
        with open(os.path.join(dir, feat_file)) as f:
            header = f.readline().rstrip()
        print_promt("The first line of the file:")
        print(header)
        return feat_file

def _add_field(field, optional=False, rec=0):
    types = ["token", "token_seq", "str", "float"]
    if optional:
        print_promt(f"Please enter the column name of `{field}`(Optional):")
    else:
        print_promt(f"Please enter the column name of `{field}`(Required):")
    field_name = print_opt()
    print_promt(f"Please select the type of `{field}`:")
    print(type_promt_rec(types, rec))
    type_input = print_opt()
    if len(type_input) == 0:
        field_type = types[0]
    else:
        field_type = types[int(type_input)]
    if len(field_name) != 0:
        return f"{field_name}:{field_type}"
    else:
        return None

def _add_config(name, type="str", optional=False, default=None):
    prompt_str = f"Please enter the {name}"
    if optional:
        prompt_str += f" (Optional,{type},"
    else:
        prompt_str += f" (Required,{type},"

    if default is None:
        prompt_str += "Default: None)"
    else:
        prompt_str += f"Default: {default})"
    print_promt(f"{prompt_str}:")
    option = print_opt()
    if len(option) == 0:
        return default
    else:
        if type=='float':
            return float(option)
        elif type == 'int':
            return int(option)
        elif type == 'bool':
            return option.lower() == 'true'
        else:
            return option



def generate_dataset_config(
        name: str,
        dir: str,
    ):
    config_file_name = f"{name}.yaml"
    config_path = os.path.join(dir, config_file_name)

    config = get_dataset_default_config('ml-100k')
    for k in config:
        config[k] = None
    config['url'] = dir

    files = os.listdir(dir)
    files_promt = "\n".join([f"\t{(i)}. {f}" for i,f in enumerate(files)])

    # inter feat
    config['inter_feat_name'] = _add_feat(dir, 'interaction', files_promt, files)
    config['inter_feat_header'] = 0

    config["user_id_field"] = _add_field("user_id_field")
    config["item_id_field"] = _add_field("item_id_field")
    config["rating_field"] = _add_field("rating_field", optional=True, rec=3)
    config["time_field"] = _add_field("time_field", optional=True, rec=3)
    if config['time_field'].split(':')[-1] == 'str':
        print_promt("Please provide the time format (e.g. {format}): ".format("%Y-%m-\%dT%H:%M:%Sz"))
        config['time_format'] = print_opt()

    # item feat
    config['item_feat_name'] = _add_feat(dir, 'item feature', files_promt, files, True)
    config['item_feat_header'] = 0
    # user feat
    config['user_feat_name'] = _add_feat(dir, 'user feature', files_promt, files, True)
    config['user_feat_header'] = 0

    config['field_separator'] = _add_config('field_separator', 'str', False, '\\t')
    config['seq_separator'] = _add_config('seq_separator', 'str', False, ' ')
    config['min_user_inter'] = _add_config('min_user_inter', 'int', True, 10)
    config['min_item_inter'] = _add_config('min_item_inter', 'int', True, 10)
    config['field_max_len'] = _add_config('field_max_len', 'int', True, None)
    config['rating_threshold'] = _add_config('rating_threshold', 'float', True, None)
    config['drop_low_rating'] = _add_config('drop_low_rating', 'bool', True, True)
    config['ranker_rating_threshold'] = _add_config('ranker_rating_threshold', 'float', True, False)
    config['max_seq_len'] = _add_config('max_seq_len', 'int', False, 20)
    config['save_cache'] = True

    with open(config_path, 'w') as f:
        documents = yaml.dump(config, f)
    print(f"Configuration file saved in {config_path}")
    print("If network features (social networks or knowledge graph) are needed, please modify\
           the configuration file manually.")
    return


# if __name__ == '__main__':
generate_dataset_config(name='my', dir='./.recstudio/tmall/')