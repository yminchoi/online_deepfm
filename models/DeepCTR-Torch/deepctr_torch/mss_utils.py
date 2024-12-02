import pandas as pd
import numpy as np
from pyspark.sql.functions import struct, col
import boto3

def get_model_params():
    return {
        'embed_dim': 8,
        'mlp_dims': [64, 32, 16],
        'dropout': 0.5,
        'learning_rate': 0.0005,
        'online_learning_rate': 0.001,
        'batch_size': 1024,
        'online_batch_size': 256,
        'epochs': 10,
        'online_epochs_day': 2,
        'online_epochs_night': 1,
        'val_ratio': 0.1
    }

def get_banner_model_configs():
    base_save_path = "banner_campaign"
    models = ["deepfm"]
    model_types = ["personalized", "segment"]

    config = {}

    for model in models:
        config[model] = {}  # Initialize the model dictionary

        for model_type in model_types:
            config[model][model_type] = {
                "save_path_prefix": f"{base_save_path}/{model}/{model_type}/continuous",
            }

    return config

def get_file_path(bucket_path, file_path, is_fullpath=False):

    obj_list = boto3.resource("s3").Bucket(bucket_path).objects.filter(Prefix=file_path)

    if is_fullpath:
        return [
            f"s3://{bucket_path}/{obj.key}"
            for obj in obj_list
            if "_SUCCESS" not in obj.key
        ]
    else:
        return [obj.key for obj in obj_list if "_SUCCESS" not in obj.key]

def get_lastest_file_path_simple(bucket_path, file_path, file_name=None, is_fullpath=True):
    if not file_path.endswith("/"):
        file_path += "/"

    # Construct the fixed lastest path
    lastest_path = f"{file_path}"

    # Build the final path
    if is_fullpath:
        if file_name:
            return f"s3://{bucket_path}/{lastest_path}/{file_name}"
        else:
            return f"s3://{bucket_path}/{lastest_path}"
    else:
        if file_name:
            return f"{lastest_path}/{file_name}"
        else:
            return lastest_path

def get_lastest_file_path_mid(bucket_path, file_path, file_name=None, is_fullpath=True):
    if not file_path.endswith("/"):
        file_path = file_path + "/"

    obj_key_list = get_file_path(bucket_path, file_path)
    obj_key_list = [x for x in obj_key_list if "$folder$" not in x]

    # dt
    lastest_dt = max(obj_key_list).split("/dt=")[1].split("/")[0]
    retun_file_path = "/".join(
        [x for x in obj_key_list if "dt=" + lastest_dt in x][0].split("/")[:-1]
    )

    # hours
    if "hour" in retun_file_path:
        lastest_hour = max(obj_key_list).split("/hour=")[1].split("/")[0]
        retun_file_path = "/".join(
            [
                x
                for x in obj_key_list
                if ("dt=" + lastest_dt in x) and ("hour=" + lastest_hour in x)
            ][0].split("/")[:-1]
        )
        print(retun_file_path)

    if is_fullpath:
        if file_name:
            return f"s3://{bucket_path}/{retun_file_path}/{file_name}"
        else:
            return f"s3://{bucket_path}/{retun_file_path}"
    else:
        if file_name:
            return f"{retun_file_path}/{file_name}"
        else:
            return f"{retun_file_path}"

def get_lastest_file_path(bucket_path, file_path, file_name=None, is_fullpath=True):
    if not file_path.endswith("/"):
        file_path = file_path + "/"

    obj_key_list = get_file_path(bucket_path, file_path)
    obj_key_list = [x for x in obj_key_list if "$folder$" not in x]

    if not obj_key_list:
        raise ValueError("No files found in the specified path.")

    # Extract date and hour from the paths
    dt_keys = [x for x in obj_key_list if "dt=" in x]
    
    if not dt_keys:
        raise ValueError("No date keys found in the specified path.")

    # Get the lastest date
    lastest_dt = max(dt_keys, key=lambda x: x.split("/dt=")[1].split("/")[0])
    lastest_dt = lastest_dt.split("/dt=")[1].split("/")[0]

    # Filter paths for the lastest date
    dt_filtered_keys = [x for x in obj_key_list if "dt=" + lastest_dt in x]

    # Get the lastest hour for the lastest date
    hour_keys = [x for x in dt_filtered_keys if "hour=" in x]
    
    if hour_keys:
        lastest_hour = max(hour_keys, key=lambda x: x.split("/hour=")[1].split("/")[0])
        lastest_hour = lastest_hour.split("/hour=")[1].split("/")[0]

        # Filter for the lastest hour
        final_key = [x for x in dt_filtered_keys if "hour=" + lastest_hour in x]
        
        if final_key:
            final_key = final_key[0]
        else:
            raise ValueError("No matching hour found for the lastest date.")
    else:
        final_key = dt_filtered_keys[0]

    # Construct the return path
    return_path = "/".join(final_key.split("/")[:])

    if is_fullpath:
        if file_name:
            return f"s3://{bucket_path}/{return_path}/{file_name}"
        else:
            return f"s3://{bucket_path}/{return_path}"
    else:
        if file_name:
            return f"{return_path}/{file_name}"
        else:
            return return_path

def feature_selection():
    print("2. feature selection")

    sparse_features = [
        #"uid", "banner_campaign_ad_group_display_type_contents_id",
        "age_band", "gender", 
        #"clk_top5_brand", "cart_top5_brand", "phs_top5_brand", 
        "clk_top_brand", "clk_top2_brand", "clk_top3_brand",
        "target_gender", "title_token", "is_human",
        "sale_kwd_yn", "new_prod_kwd_yn", "event_kwd_yn", 
        "seasonal_kwd_yn", "category_kwd_yn", "style_kwd_yn", 
        "celeb_kwd_yn","curation_kwd_yn", "gender_kwd_yn", 
        "cxt_kwd_yn", "first_kwd_yn", "popular_kwd_yn", "mss_only_kwd_yn"
        ]
    dense_features = ["ord_freq", "discount_purchase_rate"]


    print("sparse feature :", sparse_features)
    print("dense feature :", dense_features)
    print("target :", 'is_clicked')

    return sparse_features, dense_features


def feature_encoding(data, sparse_features, dense_features, encoders):
    print("4. feature encoding ")

    print("categorical value to numeric label")
    print("sparse_features :", sparse_features)
    
    for feat in sparse_features:
        lbe = encoders[feat]  # Use the loaded encoder
        
        # Create a mapping from category to label
        mapping = {category: label for label, category in enumerate(lbe.classes_)}

        if feat in sparse_features:
            # Use pandas map for known values and handle unseen labels
            data[feat] = data[feat].astype(str).map(mapping)

            # Assign the new index for unseen labels
            data[feat] = data[feat].fillna(len(lbe.classes_)).astype(int)

        else:
            # Use pandas map for fast encoding for other features
            data[feat] = data[feat].astype(str).map(mapping).fillna(len(lbe.classes_)).astype(int)

    return data
