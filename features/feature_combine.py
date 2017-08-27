# Combine features
import os
import inspect

import pandas as pd

from .utils import *
from .feature_eng import *

def list_features(feature_module):
    # Iterate through functions to generate a feature list
    feature_list = []
    functions = inspect.getmembers(feature_module, lambda x: callable(x))
    for name, function in functions:
        name_lower = name.lower()
        if 'bin' in name_lower or 'cross' in name_lower or 'encoder' in name_lower:
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

def feature_combine(
    df, feature_list, global_force_generate=False, pickle_folder='feature_pickles/'):
    features = []
    for name, generator, kwparams, pickle_path, feature_force_generate in feature_list:
        pickle_path = pickle_folder + pickle_path
        if not global_force_generate and not feature_force_generate and os.path.exists(pickle_path):
            feature = pd.read_pickle(pickle_path)
        else:
            print(pickle_path)
            feature = generator(df, **kwparams)
            feature.to_pickle(pickle_path)
        # Rename Series so that they have proper names in the resulting
        # dataframe
        if isinstance(feature, pd.Series):
            feature.rename(name, inplace=True)
        features.append(feature)
    return pd.concat([df, *features], axis=1)

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
    prop = load_properties_data(data_folder='../data/')
    features = feature_combine(prop, feature_list)
    print(features.shape)
