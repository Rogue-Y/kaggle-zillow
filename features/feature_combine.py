# Combine features
import gc
import os
import inspect

import pandas as pd

from .utils import *
# from .feature_eng import *
from features import feature_eng
from features import feature_eng_cleaned
# from .feature_clean import *

feature_eng_with_nan_dict = dict([o for o in inspect.getmembers(feature_eng) if inspect.isfunction(o[1])])
feature_eng_no_nan_dict = dict([o for o in inspect.getmembers(feature_eng_cleaned) if inspect.isfunction(o[1])])

def list_features(feature_module):
    # Iterate through functions to generate a feature list
    feature_list = []
    functions = inspect.getmembers(feature_module, lambda x: callable(x))
    for name, function in functions:
        name_lower = name.lower()
        if 'bin' in name_lower or 'cross' in name_lower or 'encoder' in name_lower or 'helper' in name_lower:
            continue
        feature_list.append(name)
    with open('feature_list.txt', 'w') as feature_list_handler:
        for function_name in feature_list:
            feature_list_handler.write(
                "('%s', %s, {}, '%s', False),\n"
                % (function_name, function_name, function_name+'_pickle'))

# each feature in the feature_list is a tuple in the form (name, method, method_params, pickle_path, force_generate)
# TODO(hzn):
#   1. Add a api for normalization or other feature transformation.
#   2. Add a api for column to drop when add a feature
#   3. Add a poly generator for the features, so when a feature is added, also
#      add its up to nth poly

# def original_feature_clean(df, feature_module, global_force_generate=True, pickle_folder='feature_pickles/'):
#     functions = [o for o in inspect.getmembers(feature_module) if inspect.isfunction(o[1])]
#     features = []
#     for name, generator in functions:
#         name_lower = name.lower()
#         if 'bin' in name_lower or 'cross' in name_lower or 'encoder' in name_lower or 'helper' in name_lower:
#             continue
#         pickle_path = pickle_folder + str(name) + '_pickle'
#         if not global_force_generate and os.path.exists(pickle_path):
#             feature = pd.read_pickle(pickle_path)
#         else:
#             print(pickle_path)
#             # Let the original feature generators have no arguments in the definition. If different cleaning methods are
#             # needed, just make a new feature.
#             feature = generator(df)
#             # Rename Series so that they have proper names in the resulting
#             # dataframe
#             if isinstance(feature, pd.Series):
#                 feature.rename(name, inplace=True)
#
#             # TODO: Currently contains strings. Need insert one-hot encodings and other preprocess.
#             feature.to_pickle(pickle_path)
#         features.append(feature)
#     return pd.concat([*features], axis=1)

# feature_list should be a dictionary with key: before_fill, original, generated,
# and values are lists of features.
# filled feature pickes are put in a different folder
def feature_combine(feature_list, clean, pickle_folder, global_force_generate):

    if clean:
        feature_eng_dict = feature_eng_no_nan_dict
    else:
        feature_eng_dict = feature_eng_with_nan_dict

    generated_features = []

    before_fill_feature_list = feature_list['before_fill'] if 'before_fill' in feature_list else []
    prop_raw = None
    for name, generator_name, kwparams, pickle_path, feature_force_generate in before_fill_feature_list:
        pickle_path = pickle_folder + pickle_path
        if not global_force_generate and not feature_force_generate and os.path.exists(pickle_path):
            feature = pd.read_pickle(pickle_path)
        else:
            print(pickle_path)
            if prop_raw is None:
                prop_raw = load_properties_data_raw(force_read=global_force_generate)
                # need to remove parcels which does not have either lat or lon,
                # to keep consistent with the following cleaned dataset.
                if clean:
                    prop_raw = prop_raw[prop_raw['latitude'].notnull() & prop_raw['longitude'].notnull()]
            feature = feature_eng_dict[generator_name](prop_raw, **kwparams)
            # Rename Series so that they have proper names in the resulting
            # dataframe
            if isinstance(feature, pd.Series):
                feature.rename(name, inplace=True)
                feature = feature.astype('float32')
            else:
                for col in feature.columns:
                    feature[col] = feature[col].astype('float32')
            if not os.path.exists(pickle_folder):
                os.makedirs(pickle_folder)
            feature.to_pickle(pickle_path)
        generated_features.append(feature)
    del prop_raw; gc.collect()

    # Clean the original features
    # TODO(hzn):
    # 1. some features like missing value related need to be generated before
    # clean
    # 2. geo fill nan may need special attention(like drop all rows where lat
    # and lon are nan)
    # 3. for feature generated by a/b, when a != 0 and b == 0, the result is
    # np.inf, when a == b == 0, the result is np.nan, need to clean this
    # 4. shrink the size of the dataset after fill na

    if clean:
        prop = load_properties_data_cleaned(force_read=global_force_generate)
    else:
        prop = load_properties_data_preprocessed(force_read=global_force_generate)

    generated_feature_list = feature_list['generated'] if 'generated' in feature_list else []
    for name, generator_name, kwparams, pickle_path, feature_force_generate in generated_feature_list:
        pickle_path = pickle_folder + pickle_path
        if not global_force_generate and not feature_force_generate and os.path.exists(pickle_path):
            feature = pd.read_pickle(pickle_path)
        else:
            print(pickle_path)
            feature = feature_eng_dict[generator_name](prop, **kwparams)
            # Rename Series so that they have proper names in the resulting
            # dataframe
            if isinstance(feature, pd.Series):
                feature.rename(name, inplace=True)
                feature = feature.astype('float32')
            else:
                for col in feature.columns:
                    feature[col] = feature[col].astype('float32')
            if not os.path.exists(pickle_folder):
                os.makedirs(pickle_folder)
            feature.to_pickle(pickle_path)
        generated_features.append(feature)

    original_features = feature_list['original']

    df = pd.concat([prop[original_features], *generated_features], axis=1)

    # Sanity check
    if clean:
        for col in df.columns:
            nan_count = df[col].isnull().sum()
            try:
                if nan_count > 0:
                    print(col + ' : ' + str(nan_count))
            except:
                print(col)

    return df

def feature_combine_cleaned(
    feature_list, global_force_generate=False, pickle_folder='features/feature_pickles_cleaned/'):

    return feature_combine(feature_list, True, pickle_folder, global_force_generate)


def feature_combine_with_nan(
    feature_list, global_force_generate=False, pickle_folder='features/feature_pickles/'):

    return feature_combine(feature_list, False, pickle_folder, global_force_generate)


if __name__ == "__main__":
    feature_list = [
        ('average_room_size', average_room_size, {}, 'average_room_size_pickle'),
        ('boolean_has_ac', boolean_has_ac, {}, 'boolean_has_ac_pickle'),
        ('boolean_has_garage_pool_or_ac', boolean_has_garage_pool_or_ac, {}, 'boolean_has_garage_pool_or_ac_pickle'),
        ('boolean_has_heat', boolean_has_heat, {}, 'boolean_has_heat_pickle'),
        ('building_age', building_age, {}, 'building_age_pickle'),
        ('built_before_year', built_before_year, {}, 'built_before_year_pickle'),
        ('category_ac_type_encode', category_ac_type_encode, {}, 'category_ac_type_encode_pickle'),
        ('category_ac_type_one_hot', category_ac_type_one_hot, {}, 'category_ac_type_one_hot_pickle'),
        ('category_fips_type_encode', category_fips_type_encode, {}, 'category_fips_type_encode_pickle'),
        ('category_fips_type_one_hot', category_fips_type_one_hot, {}, 'category_fips_type_one_hot_pickle'),
        ('category_heating_type_encode', category_heating_type_encode, {}, 'category_heating_type_encode_pickle'),
        ('category_heating_type_one_hot', category_heating_type_one_hot, {}, 'category_heating_type_one_hot_pickle'),
        ('category_land_use_code_encode', category_land_use_code_encode, {}, 'category_land_use_code_encode_pickle'),
        ('category_land_use_code_one_hot', category_land_use_code_one_hot, {}, 'category_land_use_code_one_hot_pickle'),
        ('category_land_use_desc_encode', category_land_use_desc_encode, {}, 'category_land_use_desc_encode_pickle'),
        ('category_land_use_desc_one_hot', category_land_use_desc_one_hot, {}, 'category_land_use_desc_one_hot_pickle'),
        ('category_land_use_type_encode', category_land_use_type_encode, {}, 'category_land_use_type_encode_pickle'),
        ('category_land_use_type_one_hot', category_land_use_type_one_hot, {}, 'category_land_use_type_one_hot_pickle'),
        ('deviation_from_avg_structure_tax_value', deviation_from_avg_structure_tax_value, {}, 'deviation_from_avg_structure_tax_value_pickle'),
        ('error_rate_calculated_finished_living_sqft', error_rate_calculated_finished_living_sqft, {}, 'error_rate_calculated_finished_living_sqft_pickle'),
        ('extra_rooms', extra_rooms, {}, 'extra_rooms_pickle'),
        ('extra_space', extra_space, {}, 'extra_space_pickle'),
        ('geo_city_structure_tax_value', geo_city_structure_tax_value, {}, 'geo_city_structure_tax_value_pickle'),
        ('geo_city_tax_value', geo_city_tax_value, {}, 'geo_city_tax_value_pickle'),
        ('geo_lat_lon_block', geo_lat_lon_block, {}, 'geo_lat_lon_block_pickle'),
        ('geo_lat_lon_block_tax_value', geo_lat_lon_block_tax_value, {}, 'geo_lat_lon_block_tax_value_pickle'),
        ('geo_neighorhood_tax_value', geo_neighorhood_tax_value, {}, 'geo_neighorhood_tax_value_pickle'),
        ('geo_region_tax_value', geo_region_tax_value, {}, 'geo_region_tax_value_pickle'),
        ('geo_zip_tax_value', geo_zip_tax_value, {}, 'geo_zip_tax_value_pickle'),
        ('missing_value_count', missing_value_count, {}, 'missing_value_count_pickle'),
        ('missing_value_one_hot', missing_value_one_hot, {}, 'missing_value_one_hot_pickle'),
        ('multiply_lat_lon', multiply_lat_lon, {}, 'multiply_lat_lon_pickle'),
        ('poly_2_structure_tax_value', poly_2_structure_tax_value, {}, 'poly_2_structure_tax_value_pickle'),
        ('poly_3_structure_tax_value', poly_3_structure_tax_value, {}, 'poly_3_structure_tax_value_pickle'),
        ('ratio_living_area', ratio_living_area, {}, 'ratio_living_area_pickle'),
        ('ratio_structure_tax_value_to_land_tax_value', ratio_structure_tax_value_to_land_tax_value, {}, 'ratio_structure_tax_value_to_land_tax_value_pickle'),
        ('ratio_tax', ratio_tax, {}, 'ratio_tax_pickle'),
        ('round_lat', round_lat, {}, 'round_lat_pickle'),
        ('round_lon', round_lon, {}, 'round_lon_pickle'),
        ('round_multiply_lat_lon', round_multiply_lat_lon, {}, 'round_multiply_lat_lon_pickle'),
        ('sum_lat_lon', sum_lat_lon, {}, 'sum_lat_lon_pickle'),
        ('total_rooms', total_rooms, {}, 'total_rooms_pickle')
    ]
    prop = load_properties_data_preprocessed(data_folder='../data/')
    features = feature_combine_cleaned(prop, feature_list)
    print(features.shape)
