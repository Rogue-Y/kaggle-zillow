from .feature_eng import *

# feature_list = [
#     ('average_room_size', average_room_size, {}, 'average_room_size_pickle', False),
#     ('boolean_has_ac', boolean_has_ac, {}, 'boolean_has_ac_pickle', False),
#     ('boolean_has_garage_pool_or_ac', boolean_has_garage_pool_or_ac, {}, 'boolean_has_garage_pool_or_ac_pickle', False),
#     ('boolean_has_heat', boolean_has_heat, {}, 'boolean_has_heat_pickle', False),
#     ('building_age', building_age, {}, 'building_age_pickle', False),
#     ('built_before_year', built_before_year, {}, 'built_before_year_pickle', False),
#     ('category_ac_type_encode', category_ac_type_encode, {}, 'category_ac_type_encode_pickle', False),
#     # ('category_ac_type_one_hot', category_ac_type_one_hot, {}, 'category_ac_type_one_hot_pickle', False),
#     ('category_fips_type_encode', category_fips_type_encode, {}, 'category_fips_type_encode_pickle', False),
#     # ('category_fips_type_one_hot', category_fips_type_one_hot, {}, 'category_fips_type_one_hot_pickle', False),
#     ('category_heating_type_encode', category_heating_type_encode, {}, 'category_heating_type_encode_pickle', False),
#     # ('category_heating_type_one_hot', category_heating_type_one_hot, {}, 'category_heating_type_one_hot_pickle', False),
#     # ('category_land_use_code', category_land_use_code, {}, 'category_land_use_code_pickle', False),
#     ('category_land_use_code_encode', category_land_use_code_encode, {}, 'category_land_use_code_encode_pickle', False),
#     # ('category_land_use_code_one_hot', category_land_use_code_one_hot, {}, 'category_land_use_code_one_hot_pickle', False),
#     # ('category_land_use_desc', category_land_use_desc, {}, 'category_land_use_desc_pickle', False),
#     ('category_land_use_desc_encode', category_land_use_desc_encode, {}, 'category_land_use_desc_encode_pickle', False),
#     # ('category_land_use_desc_one_hot', category_land_use_desc_one_hot, {}, 'category_land_use_desc_one_hot_pickle', False),
#     # ('category_land_use_type', category_land_use_type, {}, 'category_land_use_type_pickle', False),
#     ('category_land_use_type_encode', category_land_use_type_encode, {}, 'category_land_use_type_encode_pickle', False),
#     # ('category_land_use_type_one_hot', category_land_use_type_one_hot, {}, 'category_land_use_type_one_hot_pickle', False),
#     # ('deviation_from_avg_structure_tax_value', deviation_from_avg_structure_tax_value, {}, 'deviation_from_avg_structure_tax_value_pickle', False),
#     ('error_rate_calculated_finished_living_sqft', error_rate_calculated_finished_living_sqft, {}, 'error_rate_calculated_finished_living_sqft_pickle', False),
#     ('extra_rooms', extra_rooms, {}, 'extra_rooms_pickle', False),
#     ('extra_space', extra_space, {}, 'extra_space_pickle', False),
#     ('geo_city', geo_city, {}, 'geo_city_pickle', False),
#     # ('geo_city_structure_tax_value', geo_city_structure_tax_value, {}, 'geo_city_structure_tax_value_pickle', False),
#     # ('geo_city_tax_value', geo_city_tax_value, {}, 'geo_city_tax_value_pickle', False),
#     ('geo_county', geo_county, {}, 'geo_county_pickle', False),
#     # ('geo_lat_lon_block', geo_lat_lon_block, {}, 'geo_lat_lon_block_pickle', False),
#     # ('geo_lat_lon_block_tax_value', geo_lat_lon_block_tax_value, {}, 'geo_lat_lon_block_tax_value_pickle', False),
#     ('geo_neighborhood', geo_neighborhood, {}, 'geo_neighborhood_pickle', False),
#     ('geo_lat_lon_block_features', geo_lat_lon_block_features, {}, 'geo_lat_lon_block_features_pickle', False),
#     # ('geo_neighborhood_tax_value', geo_neighborhood_tax_value, {}, 'geo_neighborhood_tax_value_pickle', False),
#     # ('geo_neighborhood_tax_value_ratio_mean', geo_neighborhood_tax_value_ratio_mean, {}, 'geo_neighborhood_tax_value_ratio_mean_pickle', False),
#     ('geo_region_tax_value', geo_region_tax_value, {}, 'geo_region_tax_value_pickle', False),
#     ('geo_zip', geo_zip, {}, 'geo_zip_pickle', False),
#     # ('geo_zip_tax_value', geo_zip_tax_value, {}, 'geo_zip_tax_value_pickle', False),
#     ('missing_value_count', missing_value_count, {}, 'missing_value_count_pickle', False),
#     # ('missing_value_one_hot', missing_value_one_hot, {}, 'missing_value_one_hot_pickle', False),
#     ('multiply_lat_lon', multiply_lat_lon, {}, 'multiply_lat_lon_pickle', False),
#     ('poly_2_structure_tax_value', poly_2_structure_tax_value, {}, 'poly_2_structure_tax_value_pickle', False),
#     ('poly_3_structure_tax_value', poly_3_structure_tax_value, {}, 'poly_3_structure_tax_value_pickle', False),
#     ('ratio_living_area', ratio_living_area, {}, 'ratio_living_area_pickle', False),
#     ('ratio_structure_tax_value_to_land_tax_value', ratio_structure_tax_value_to_land_tax_value, {}, 'ratio_structure_tax_value_to_land_tax_value_pickle', False),
#     ('ratio_tax_value_to_structure_value', ratio_tax_value_to_structure_value, {}, 'ratio_tax_value_to_structure_value_pickle', False),
#     ('ratio_tax_value_to_land_tax_value', ratio_tax_value_to_land_tax_value, {}, 'ratio_tax_value_to_land_tax_value_pickle', False),
#     ('ratio_tax', ratio_tax, {}, 'ratio_tax_pickle', False),
#     ('round_lat', round_lat, {}, 'round_lat_pickle', False),
#     ('round_lon', round_lon, {}, 'round_lon_pickle', False),
#     ('round_multiply_lat_lon', round_multiply_lat_lon, {}, 'round_multiply_lat_lon_pickle', False),
#     ('sum_lat_lon', sum_lat_lon, {}, 'sum_lat_lon_pickle', False),
#     ('total_rooms', total_rooms, {}, 'total_rooms_pickle', False),
# ]

feature_list = [
    # ('average_room_size', average_room_size, {}, 'average_room_size_pickle', False),
    # ('boolean_has_ac', boolean_has_ac, {}, 'boolean_has_ac_pickle', False),
    # ('boolean_has_garage_pool_or_ac', boolean_has_garage_pool_or_ac, {}, 'boolean_has_garage_pool_or_ac_pickle', False),
    # ('boolean_has_heat', boolean_has_heat, {}, 'boolean_has_heat_pickle', False),
    # ('building_age', building_age, {}, 'building_age_pickle', False),
    # ('built_before_year', built_before_year, {}, 'built_before_year_pickle', False),
    # ('category_ac_type_encode', category_ac_type_encode, {}, 'category_ac_type_encode_pickle', False),
    # ('category_ac_type_one_hot', category_ac_type_one_hot, {}, 'category_ac_type_one_hot_pickle', False),
    # ('category_fips_type_encode', category_fips_type_encode, {}, 'category_fips_type_encode_pickle', False),
    # ('category_fips_type_one_hot', category_fips_type_one_hot, {}, 'category_fips_type_one_hot_pickle', False),
    # ('category_heating_type_encode', category_heating_type_encode, {}, 'category_heating_type_encode_pickle', False),
    # ('category_heating_type_one_hot', category_heating_type_one_hot, {}, 'category_heating_type_one_hot_pickle', False),
    # ('category_land_use_code', category_land_use_code, {}, 'category_land_use_code_pickle', False),
    # ('category_land_use_code_encode', category_land_use_code_encode, {}, 'category_land_use_code_encode_pickle', False),
    # ('category_land_use_code_one_hot', category_land_use_code_one_hot, {}, 'category_land_use_code_one_hot_pickle', False),
    # ('category_land_use_desc', category_land_use_desc, {}, 'category_land_use_desc_pickle', False),
    # ('category_land_use_desc_encode', category_land_use_desc_encode, {}, 'category_land_use_desc_encode_pickle', False),
    # ('category_land_use_desc_one_hot', category_land_use_desc_one_hot, {}, 'category_land_use_desc_one_hot_pickle', False),
    # ('category_land_use_type', category_land_use_type, {}, 'category_land_use_type_pickle', False),
    # ('category_land_use_type_encode', category_land_use_type_encode, {}, 'category_land_use_type_encode_pickle', False),
    # ('category_land_use_type_one_hot', category_land_use_type_one_hot, {}, 'category_land_use_type_one_hot_pickle', False),
    # ('deviation_from_avg_structure_tax_value', deviation_from_avg_structure_tax_value, {}, 'deviation_from_avg_structure_tax_value_pickle', False),
    # ('error_rate_calculated_finished_living_sqft', error_rate_calculated_finished_living_sqft, {}, 'error_rate_calculated_finished_living_sqft_pickle', False),
    # ('extra_rooms', extra_rooms, {}, 'extra_rooms_pickle', False),
    # ('extra_space', extra_space, {}, 'extra_space_pickle', False),
    # ('geo_city', geo_city, {}, 'geo_city_pickle', False),
    # ('geo_city_structure_tax_value', geo_city_structure_tax_value, {}, 'geo_city_structure_tax_value_pickle', False),
    # ('geo_city_tax_value', geo_city_tax_value, {}, 'geo_city_tax_value_pickle', False),
    # ('geo_county', geo_county, {}, 'geo_county_pickle', False),
    # ('geo_lat_lon_block', geo_lat_lon_block, {}, 'geo_lat_lon_block_pickle', False),
    # ('geo_lat_lon_block_tax_value', geo_lat_lon_block_tax_value, {}, 'geo_lat_lon_block_tax_value_pickle', False),
    ('geo_neighborhood', geo_neighborhood, {}, 'geo_neighborhood_pickle', False),
    # ('geo_lat_lon_block_features', geo_lat_lon_block_features, {}, 'geo_lat_lon_block_features_pickle', False),
    # ('geo_neighborhood_tax_value', geo_neighborhood_tax_value, {}, 'geo_neighborhood_tax_value_pickle', False),
    # ('geo_neighborhood_tax_value_ratio_mean', geo_neighborhood_tax_value_ratio_mean, {}, 'geo_neighborhood_tax_value_ratio_mean_pickle', False),
    # ('geo_region_tax_value', geo_region_tax_value, {}, 'geo_region_tax_value_pickle', False),
    ('geo_zip', geo_zip, {}, 'geo_zip_pickle', False),
    # ('geo_zip_tax_value', geo_zip_tax_value, {}, 'geo_zip_tax_value_pickle', False),
    # ('missing_value_count', missing_value_count, {}, 'missing_value_count_pickle', False),
    # ('missing_value_one_hot', missing_value_one_hot, {}, 'missing_value_one_hot_pickle', False),
    # ('multiply_lat_lon', multiply_lat_lon, {}, 'multiply_lat_lon_pickle', False),
    # ('poly_2_structure_tax_value', poly_2_structure_tax_value, {}, 'poly_2_structure_tax_value_pickle', False),
    # ('poly_3_structure_tax_value', poly_3_structure_tax_value, {}, 'poly_3_structure_tax_value_pickle', False),
    # ('ratio_living_area', ratio_living_area, {}, 'ratio_living_area_pickle', False),
    ('ratio_structure_tax_value_to_land_tax_value', ratio_structure_tax_value_to_land_tax_value, {}, 'ratio_structure_tax_value_to_land_tax_value_pickle', False),
    ('ratio_tax_value_to_structure_value', ratio_tax_value_to_structure_value, {}, 'ratio_tax_value_to_structure_value_pickle', False),
    ('ratio_tax_value_to_land_tax_value', ratio_tax_value_to_land_tax_value, {}, 'ratio_tax_value_to_land_tax_value_pickle', False),
    # ('ratio_tax', ratio_tax, {}, 'ratio_tax_pickle', False),
    # ('round_lat', round_lat, {}, 'round_lat_pickle', False),
    # ('round_lon', round_lon, {}, 'round_lon_pickle', False),
    # ('round_multiply_lat_lon', round_multiply_lat_lon, {}, 'round_multiply_lat_lon_pickle', False),
    # ('sum_lat_lon', sum_lat_lon, {}, 'sum_lat_lon_pickle', False),

    ('target_neighborhood_feature', target_region_feature, {'id_name':'regionidneighborhood'}, 'target_neighborhood_feature_pickle', False),
    ('target_zip_feature', target_region_feature, {'id_name':'regionidzip'}, 'target_zip_feature_pickle', False),
    ('target_city_feature', target_region_feature, {'id_name':'regionidcity'}, 'target_city_feature_pickle', False),
    ('target_county_feature', target_region_feature, {'id_name':'regionidcounty'}, 'target_county_feature_pickle', False),
    #
    # ('total_rooms', total_rooms, {}, 'total_rooms_pickle', False),
]

# feature_list = [
#     ('average_bathroom_size', average_bathroom_size, {}, 'average_bathroom_size_pickle', False),
#     ('average_bedroom_size', average_bedroom_size, {}, 'average_bedroom_size_pickle', False),
#     ('average_pool_size', average_pool_size, {}, 'average_pool_size_pickle', False),
#     ('average_room_size', average_room_size, {}, 'average_room_size_pickle', False),
#     ('average_room_size_2', average_room_size_2, {}, 'average_room_size_2_pickle', False),
#     ('boolean_has_ac', boolean_has_ac, {}, 'boolean_has_ac_pickle', False),
#     ('boolean_has_garage_pool_or_ac', boolean_has_garage_pool_or_ac, {}, 'boolean_has_garage_pool_or_ac_pickle', False),
#     ('boolean_has_heat', boolean_has_heat, {}, 'boolean_has_heat_pickle', False),
#     ('building_age', building_age, {}, 'building_age_pickle', False),
#     ('built_before_year', built_before_year, {}, 'built_before_year_pickle', False),
#     # ('category_ac_type_encode', category_ac_type_encode, {}, 'category_ac_type_encode_pickle', False),
#     # ('category_ac_type_one_hot', category_ac_type_one_hot, {}, 'category_ac_type_one_hot_pickle', False),
#     # ('category_fips_type_encode', category_fips_type_encode, {}, 'category_fips_type_encode_pickle', False),
#     # ('category_fips_type_one_hot', category_fips_type_one_hot, {}, 'category_fips_type_one_hot_pickle', False),
#     # ('category_heating_type_encode', category_heating_type_encode, {}, 'category_heating_type_encode_pickle', False),
#     # ('category_heating_type_one_hot', category_heating_type_one_hot, {}, 'category_heating_type_one_hot_pickle', False),
#     # ('category_land_use_code', category_land_use_code, {}, 'category_land_use_code_pickle', False),
#     ('category_land_use_code_encode', category_land_use_code_encode, {}, 'category_land_use_code_encode_pickle', False),
#     # ('category_land_use_code_one_hot', category_land_use_code_one_hot, {}, 'category_land_use_code_one_hot_pickle', False),
#     # ('category_land_use_desc', category_land_use_desc, {}, 'category_land_use_desc_pickle', False),
#     ('category_land_use_desc_encode', category_land_use_desc_encode, {}, 'category_land_use_desc_encode_pickle', False),
#     # ('category_land_use_desc_one_hot', category_land_use_desc_one_hot, {}, 'category_land_use_desc_one_hot_pickle', False),
#     # ('category_land_use_type', category_land_use_type, {}, 'category_land_use_type_pickle', False),
#     ('category_land_use_type_encode', category_land_use_type_encode, {}, 'category_land_use_type_encode_pickle', False),
#     # ('category_land_use_type_one_hot', category_land_use_type_one_hot, {}, 'category_land_use_type_one_hot_pickle', False),
#     ('error_rate_bathroom', error_rate_bathroom, {}, 'error_rate_bathroom_pickle', False),
#     ('error_rate_calculated_finished_living_sqft', error_rate_calculated_finished_living_sqft, {}, 'error_rate_calculated_finished_living_sqft_pickle', False),
#     ('error_rate_count_bathroom', error_rate_count_bathroom, {}, 'error_rate_count_bathroom_pickle', False),
#     ('error_rate_first_floor_living_sqft', error_rate_first_floor_living_sqft, {}, 'error_rate_first_floor_living_sqft_pickle', False),
#     ('extra_rooms', extra_rooms, {}, 'extra_rooms_pickle', False),
#     ('extra_space', extra_space, {}, 'extra_space_pickle', False),
#     ('geo_city', geo_city, {}, 'geo_city_pickle', False),
#     ('geo_county', geo_county, {}, 'geo_county_pickle', False),
#     ('geo_lat_lon_block_features', geo_lat_lon_block_features, {}, 'geo_lat_lon_block_features_pickle', False),
#     ('geo_neighborhood', geo_neighborhood, {}, 'geo_neighborhood_pickle', False),
#     ('geo_zip', geo_zip, {}, 'geo_zip_pickle', False),
#     ('missing_value_count', missing_value_count, {}, 'missing_value_count_pickle', False),
#     # ('missing_value_one_hot', missing_value_one_hot, {}, 'missing_value_one_hot_pickle', False),
#     ('multiply_lat_lon', multiply_lat_lon, {}, 'multiply_lat_lon_pickle', False),
#     ('poly_2_structure_tax_value', poly_2_structure_tax_value, {}, 'poly_2_structure_tax_value_pickle', False),
#     ('poly_3_structure_tax_value', poly_3_structure_tax_value, {}, 'poly_3_structure_tax_value_pickle', False),
#     ('ratio_basement', ratio_basement, {}, 'ratio_basement_pickle', False),
#     ('ratio_bedroom_bathroom', ratio_bedroom_bathroom, {}, 'ratio_bedroom_bathroom_pickle', False),
#     ('ratio_fireplace', ratio_fireplace, {}, 'ratio_fireplace_pickle', False),
#     ('ratio_floor_shape', ratio_floor_shape, {}, 'ratio_floor_shape_pickle', False),
#     ('ratio_living_area', ratio_living_area, {}, 'ratio_living_area_pickle', False),
#     ('ratio_living_area_2', ratio_living_area_2, {}, 'ratio_living_area_2_pickle', False),
#     ('ratio_pool_shed', ratio_pool_shed, {}, 'ratio_pool_shed_pickle', False),
#     ('ratio_pool_yard', ratio_pool_yard, {}, 'ratio_pool_yard_pickle', False),
#     ('ratio_structure_tax_value_to_land_tax_value', ratio_structure_tax_value_to_land_tax_value, {}, 'ratio_structure_tax_value_to_land_tax_value_pickle', False),
#     ('ratio_tax', ratio_tax, {}, 'ratio_tax_pickle', False),
#     ('ratio_tax_value_to_land_tax_value', ratio_tax_value_to_land_tax_value, {}, 'ratio_tax_value_to_land_tax_value_pickle', False),
#     ('ratio_tax_value_to_structure_value', ratio_tax_value_to_structure_value, {}, 'ratio_tax_value_to_structure_value_pickle', False),
#     ('round_lat', round_lat, {}, 'round_lat_pickle', False),
#     ('round_lon', round_lon, {}, 'round_lon_pickle', False),
#     # ('round_multiply_lat_lon', round_multiply_lat_lon, {}, 'round_multiply_lat_lon_pickle', False),
#     ('sum_lat_lon', sum_lat_lon, {}, 'sum_lat_lon_pickle', False),
#     ('total_rooms', total_rooms, {}, 'total_rooms_pickle', False),
#
#     # ('target_neighborhood_feature', target_region_feature, {'id_name':'regionidneighborhood'}, 'target_neighborhood_feature_pickle', False),
#     # ('target_zip_feature', target_region_feature, {'id_name':'regionidzip'}, 'target_zip_feature_pickle', False),
#     # ('target_city_feature', target_region_feature, {'id_name':'regionidcity'}, 'target_city_feature_pickle', False),
#     # ('target_county_feature', target_region_feature, {'id_name':'regionidcounty'}, 'target_county_feature_pickle', False),
# ]